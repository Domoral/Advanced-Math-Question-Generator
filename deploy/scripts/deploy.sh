#!/bin/bash

# 高数题目生成器部署脚本

set -e

echo "========================================"
echo "  高数题目生成器部署脚本"
echo "========================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_ROOT"

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 1. 构建前端
echo -e "${YELLOW}[步骤 1/4] 构建前端项目...${NC}"
cd frontend

if ! command_exists npm; then
    echo -e "${RED}错误: 未找到 npm，请先安装 Node.js${NC}"
    exit 1
fi

echo "安装依赖..."
npm install

echo "构建生产版本..."
npm run build

cd ..
echo -e "${GREEN}前端构建完成!${NC}"

# 2. 检查 Nginx
echo -e "${YELLOW}[步骤 2/4] 检查 Nginx...${NC}"

if command_exists nginx; then
    echo -e "${GREEN}检测到系统已安装 Nginx${NC}"
    INSTALL_METHOD="system"
elif command_exists docker; then
    echo -e "${GREEN}检测到 Docker，将使用 Docker 部署 Nginx${NC}"
    INSTALL_METHOD="docker"
else
    echo -e "${YELLOW}未检测到 Nginx 或 Docker${NC}"
    echo "请选择安装方式:"
    echo "1) 使用 Docker 部署 (推荐)"
    echo "2) 手动安装 Nginx"
    read -p "请输入选项 (1/2): " choice

    case $choice in
        1)
            echo "正在安装 Docker..."
            curl -fsSL https://get.docker.com | sh
            sudo usermod -aG docker $USER
            INSTALL_METHOD="docker"
            ;;
        2)
            echo "请手动安装 Nginx 后重试"
            echo "Ubuntu/Debian: sudo apt install nginx"
            echo "CentOS/RHEL: sudo yum install nginx"
            exit 0
            ;;
        *)
            echo -e "${RED}无效选项${NC}"
            exit 1
            ;;
    esac
fi

# 3. 部署
echo -e "${YELLOW}[步骤 3/4] 部署到 Nginx...${NC}"

if [ "$INSTALL_METHOD" = "system" ]; then
    # 系统 Nginx 部署
    echo "使用系统 Nginx 部署..."

    # 复制构建文件
    sudo rm -rf /var/www/math-question-generator
    sudo mkdir -p /var/www/math-question-generator
    sudo cp -r frontend/build/* /var/www/math-question-generator/

    # 复制 Nginx 配置
    sudo cp deploy/nginx/nginx.conf /etc/nginx/sites-available/math-question-generator

    # 启用站点
    if [ ! -L /etc/nginx/sites-enabled/math-question-generator ]; then
        sudo ln -s /etc/nginx/sites-available/math-question-generator /etc/nginx/sites-enabled/
    fi

    # 测试配置
    sudo nginx -t

    # 重启 Nginx
    sudo systemctl restart nginx

    echo -e "${GREEN}系统 Nginx 部署完成!${NC}"

elif [ "$INSTALL_METHOD" = "docker" ]; then
    # Docker 部署
    echo "使用 Docker 部署..."

    cd deploy/docker
    if command_exists docker-compose; then
        docker-compose up -d --build
    else
        docker compose up -d --build
    fi
    cd ../..

    echo -e "${GREEN}Docker 部署完成!${NC}"
fi

# 4. 完成
echo -e "${YELLOW}[步骤 4/4] 验证部署...${NC}"

sleep 2

if curl -s -o /dev/null -w "%{http_code}" http://localhost | grep -q "200"; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  部署成功!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "访问地址:"
    echo "  - 本地: http://localhost"
    echo "  - 局域网: http://$(hostname -I | awk '{print $1}')"
    echo ""
else
    echo -e "${RED}部署可能出现问题，请检查日志${NC}"
fi

echo ""
echo "常用命令:"
if [ "$INSTALL_METHOD" = "docker" ]; then
    echo "  查看日志: docker logs math-question-nginx"
    echo "  停止服务: cd deploy/docker && docker-compose down"
    echo "  重启服务: cd deploy/docker && docker-compose restart"
else
    echo "  查看日志: sudo tail -f /var/log/nginx/access.log"
    echo "  停止服务: sudo systemctl stop nginx"
    echo "  重启服务: sudo systemctl restart nginx"
fi
