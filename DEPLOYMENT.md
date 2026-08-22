# 服务器部署流程

## 部署边界

- 本地负责构建镜像、打版本标签并推送到阿里云镜像仓库。
- 服务器不构建代码，每次发布只拉取指定版本镜像并重建容器。
- 阿里云镜像仓库已登录，不重复执行 `docker login`。
- 服务器连接方式以公司环境资料为准，不在仓库中记录账号、密钥或密码。

## 前置条件

服务器部署目录固定为 `/opt/data-agent`，需要提前准备好以下文件：

```text
/opt/data-agent/
├── docker-compose.yml
└── configs/
    └── config.yaml
```

`configs/config.yaml` 由服务器本地维护真实配置，不应把生产密钥打进镜像。
`docker-compose.yml` 会将该文件只读挂载到容器内。
首次部署时从 `configs/config.example.yaml` 复制生成，并填写目标环境配置；
`configs/config.yaml` 已被 Git 忽略。

Dockerfile 只复制应用代码和依赖声明，不会把 `configs/config.yaml` 打进镜像。

## 一、本地构建镜像

镜像版本号必须唯一且不可覆盖。推荐使用“日期时间 + Git 短提交号”，例如：

```bash
export IMAGE_VERSION=20260821-359ecd3
export LOCAL_IMAGE=data-agen:${IMAGE_VERSION}
export REMOTE_IMAGE=registry.cn-hangzhou.aliyuncs.com/yinzixiang/yzx:${IMAGE_VERSION}
```

目标服务器使用 Linux AMD64 镜像。本地如果是 Apple Silicon，使用：

```bash
docker buildx build \
  --platform linux/amd64 \
  --load \
  --tag "${LOCAL_IMAGE}" \
  .
```

确认镜像已经生成：

```bash
docker image inspect "${LOCAL_IMAGE}"
```

## 二、打标签并推送

使用本地镜像名称打标签：

```bash
docker tag "${LOCAL_IMAGE}" "${REMOTE_IMAGE}"
docker push "${REMOTE_IMAGE}"
```

如果已经拿到镜像 ID，也可以直接使用：

```bash
docker tag [ImageId] registry.cn-hangzhou.aliyuncs.com/yinzixiang/yzx:[镜像版本号]
docker push registry.cn-hangzhou.aliyuncs.com/yinzixiang/yzx:[镜像版本号]
```

推送完成后记录本次 `IMAGE_VERSION`，服务器部署和回滚都使用这个不可变版本号。

## 三、服务器拉取并部署

进入服务器已有的部署目录，然后设置与本地推送一致的版本号：

```bash
cd /opt/data-agent

export IMAGE_VERSION=20260821-359ecd3
export DATA_AGENT_IMAGE=registry.cn-hangzhou.aliyuncs.com/yinzixiang/yzx:${IMAGE_VERSION}
```

拉取镜像并确认 Compose 最终配置：

```bash
docker pull "${DATA_AGENT_IMAGE}"
docker compose config
```

重建并启动容器：

```bash
docker compose up -d --remove-orphans
```

## 四、部署验证

```bash
docker compose ps
docker compose logs --tail=200 data-agen
curl -fsS http://127.0.0.1:9090/health
```

只有容器状态正常且 `/health` 返回成功后，才视为本次发布完成。

## 五、版本回滚

将 `DATA_AGENT_IMAGE` 改成上一个已验证版本，重新拉取并启动：

```bash
export IMAGE_VERSION=<上一个稳定版本号>
export DATA_AGENT_IMAGE=registry.cn-hangzhou.aliyuncs.com/yinzixiang/yzx:${IMAGE_VERSION}

docker pull "${DATA_AGENT_IMAGE}"
docker compose up -d --remove-orphans
curl -fsS http://127.0.0.1:9090/health
```

不要使用同名标签覆盖旧镜像，否则无法可靠回滚。
