"""Nacos 服务注册。

把 AI 客服服务注册到 Nacos，使 Spring Cloud Gateway 能通过
``lb://tenhub-ai-service`` 发现并路由到本服务，与其它 Java 微服务保持一致。

关键设计
--------
1. **必须显式指定 IP**。本机存在 WSL / Hyper-V 等多个虚拟网卡，
   Nacos SDK 自动探测出的 IP 往往是 ``172.17.x.x`` 这类虚拟网卡地址，
   网关按该地址回调会连接失败。因此建议显式配置 ``nacos_ip``，
   并与其它 Java 服务注册的 IP 一致（本项目实测 tenhub-member 用的是
   172.29.94.120）。

2. **命名空间默认 public**。实测本项目中网关的服务发现使用 public 命名空间
   （gateway 的 ``spring.cloud.nacos.config.namespace`` 只作用于配置中心），
   ``tenhub-member`` 也注册在 public。若注册到其它命名空间，网关切不到。

3. **仅注册实例，不拉取配置**。本服务配置走本地 ``.env``。

4. 注册失败**不阻断服务启动**：Nacos 不可用时服务仍可作为独立 HTTP 服务运行，
   只是网关无法发现它。失败原因会写日志并通过 ``/health`` 暴露。

SDK 版本适配
------------
``nacos-sdk-python`` 3.x 的导入名与 API 与 2.x 完全不同：

* 3.x：``from v2.nacos import NacosNamingService, ClientConfig, ...``，**全异步**；
* 2.x 及更早：``import nacos; nacos.NacosClient(...)``，**同步**。

两种形态都在下面做了适配，避免因依赖版本不同导致注册静默失效。
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings, settings as default_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# 注册到 Nacos 的实例元数据，便于在控制台识别来源
INSTANCE_METADATA_KEYS = ("service", "framework", "purpose", "group")


def detect_local_ip() -> str:
    """探测本机局域网 IP（仅在未显式配置时使用）。"""
    try:
        # 不真正发包，只让内核选出到外网的路由对应网卡
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(2)
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"


@dataclass
class NacosRegistration:
    """一次服务注册的结果。"""

    service_name: str
    ip: str
    port: int
    namespace: str
    group: str
    registered: bool = False
    error: str = ""
    # 实际使用的 SDK 形态：v3-async / legacy-sync
    sdk: str = ""

    def describe(self) -> str:
        ns = self.namespace or "public"
        if self.registered:
            return (
                f"{self.service_name} -> {self.ip}:{self.port} "
                f"(namespace={ns}, group={self.group}, sdk={self.sdk})"
            )
        return f"{self.service_name} 未注册（{self.error or '未启用'}）"


class NacosRegistrar:
    """Nacos 服务注册管理器（自动适配 SDK 3.x 异步与 2.x 同步两种 API）。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
        self._client: Any = None
        self._mode: str = ""
        self._registration: NacosRegistration | None = None

    @property
    def registration(self) -> NacosRegistration | None:
        return self._registration

    # ------------------------------------------------------------------
    # 构造客户端
    # ------------------------------------------------------------------
    def _build_client(self) -> tuple[Any, str]:
        """返回 ``(客户端, 模式)``，模式为 ``v3-config`` 或 ``legacy``。"""
        cfg = self.settings

        # --- 优先尝试 3.x（gRPC，异步）---
        # 直接用 ClientConfig 而非 ClientConfigBuilder：builder 的方法名易变
        # （例如命名空间是 namespace_id() 而不是 namespace()），
        # 直接构造数据类更稳定。
        try:
            from v2.nacos import ClientConfig

            client_config = ClientConfig(
                server_addresses=cfg.nacos_server_addr,
                namespace_id=cfg.nacos_namespace or "",
                username=cfg.nacos_username or None,
                password=cfg.nacos_password or None,
                log_level="WARN",
            )
            return client_config, "v3-config"
        except ImportError:
            logger.debug("未找到 v2.nacos（SDK 3.x），尝试 legacy API")
        except Exception as exc:  # noqa: BLE001
            logger.warning("构造 SDK 3.x ClientConfig 失败: %s", exc)

        # --- 回退 2.x（HTTP，同步）---
        try:
            import nacos  # type: ignore[import-not-found]

            common: dict[str, Any] = {
                "server_addresses": cfg.nacos_server_addr,
                "namespace": cfg.nacos_namespace,
            }
            if cfg.nacos_username and cfg.nacos_password:
                common["username"] = cfg.nacos_username
                common["password"] = cfg.nacos_password
            return nacos.NacosClient(**common), "legacy"
        except ImportError as exc:
            raise RuntimeError(
                "未安装可用的 Nacos SDK。请执行：pip install nacos-sdk-python"
            ) from exc

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------
    async def register(self) -> NacosRegistration:
        """注册服务实例。

        Returns:
            注册结果。失败时 ``registered=False`` 且 ``error`` 记录原因，
            **不抛异常**，以免阻断服务启动。
        """
        cfg = self.settings

        if not cfg.nacos_enabled:
            self._registration = NacosRegistration(
                service_name=cfg.nacos_service_name,
                ip="",
                port=0,
                namespace=cfg.nacos_namespace,
                group=cfg.nacos_group,
                error="已通过 AI_CS_NACOS_ENABLED=false 关闭",
            )
            logger.info("已跳过 Nacos 注册（配置关闭）")
            return self._registration

        ip = cfg.nacos_ip.strip() or detect_local_ip()
        if not cfg.nacos_ip.strip():
            logger.warning(
                "未配置 AI_CS_NACOS_IP，自动探测到 %s。"
                "本机多网卡环境下该地址可能是虚拟网卡，网关可能无法回调，建议显式指定。",
                ip,
            )

        result = NacosRegistration(
            service_name=cfg.nacos_service_name,
            ip=ip,
            port=cfg.nacos_register_port,
            namespace=cfg.nacos_namespace,
            group=cfg.nacos_group,
        )
        self._registration = result

        try:
            client, mode = self._build_client()
        except Exception as exc:  # noqa: BLE001
            result.error = f"{type(exc).__name__}: {exc}"
            logger.error("Nacos 客户端初始化失败: %s", result.error)
            return result

        metadata = {
            "service": cfg.app_name,
            "framework": "fastapi",
            "purpose": "ai-customer-service",
        }

        try:
            if mode == "v3-config":
                await self._register_v3(client, ip, metadata)
                result.sdk = "v3-async"
            else:
                self._register_legacy(client, ip, metadata)
                result.sdk = "legacy-sync"

            result.registered = True
            logger.info("Nacos 注册成功: %s", result.describe())
        except Exception as exc:  # noqa: BLE001 - 注册失败不阻断启动
            result.error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "Nacos 注册失败（服务仍会启动，但网关无法发现本服务）: %s", result.error
            )

        return result

    async def _register_v3(self, client_config: Any, ip: str, metadata: dict[str, str]) -> None:
        """SDK 3.x：异步 gRPC 命名服务。"""
        from v2.nacos import NacosNamingService, RegisterInstanceParam

        cfg = self.settings
        naming = await NacosNamingService.create_naming_service(client_config)
        self._client = naming

        ok = await naming.register_instance(
            RegisterInstanceParam(
                service_name=cfg.nacos_service_name,
                group_name=cfg.nacos_group,
                ip=ip,
                port=cfg.nacos_register_port,
                weight=1.0,
                enabled=True,
                healthy=True,
                ephemeral=True,
                cluster_name="DEFAULT",
                metadata=metadata,
            )
        )
        if not ok:
            raise RuntimeError("register_instance 返回 False")

    def _register_legacy(self, client: Any, ip: str, metadata: dict[str, str]) -> None:
        """SDK 2.x 及更早：同步 HTTP 命名服务。"""
        cfg = self.settings
        self._client = client
        client.add_naming_instance(
            service_name=cfg.nacos_service_name,
            ip=ip,
            port=cfg.nacos_register_port,
            group_name=cfg.nacos_group,
            cluster_name="DEFAULT",
            weight=1.0,
            metadata=metadata,
            enable=True,
            healthy=True,
            ephemeral=True,
        )

    # ------------------------------------------------------------------
    # 注销
    # ------------------------------------------------------------------
    async def deregister(self) -> None:
        """注销服务实例（服务关闭时调用）。"""
        reg = self._registration
        if not reg or not reg.registered or self._client is None:
            return

        cfg = self.settings
        try:
            if reg.sdk == "v3-async":
                from v2.nacos import DeregisterInstanceParam

                await self._client.deregister_instance(
                    DeregisterInstanceParam(
                        service_name=reg.service_name,
                        group_name=reg.group,
                        ip=reg.ip,
                        port=reg.port,
                        cluster_name="DEFAULT",
                        ephemeral=True,
                    )
                )
                await self._client.shutdown()
            else:
                self._client.remove_naming_instance(
                    service_name=reg.service_name,
                    ip=reg.ip,
                    port=reg.port,
                    group_name=reg.group,
                    cluster_name="DEFAULT",
                    ephemeral=True,
                )
            logger.info("已从 Nacos 注销: %s", reg.describe())
        except Exception as exc:  # noqa: BLE001 - 注销失败无需影响退出
            logger.warning("Nacos 注销失败（实例可能需等心跳超时自动剔除）: %s", exc)
        finally:
            reg.registered = False


_registrar: NacosRegistrar | None = None


def get_registrar() -> NacosRegistrar:
    """获取全局注册管理器。"""
    global _registrar
    if _registrar is None:
        _registrar = NacosRegistrar()
    return _registrar
