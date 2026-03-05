# 在 init_exchange 函数中添加更详细的错误处理
@st.cache_resource
def init_exchange(exchange_name):
    """初始化交易所客户端，并提供友好错误提示"""
    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},  # 期货
            'timeout': 30000
        })
        # 测试连接
        exchange.fetch_time()
        return exchange
    except ccxt.AuthenticationError:
        st.error("🔑 API 密钥错误或需要认证（公开数据无需密钥）")
    except ccxt.BadSymbol:
        st.error(f"❌ 交易对 {symbol} 不存在，请检查格式（如 ETH/USDT, ETH-USDT）")
    except ccxt.RequestTimeout:
        st.error("⏱️ 连接超时，请检查网络")
    except Exception as e:
        error_msg = str(e)
        if "restricted location" in error_msg.lower():
            st.error("🌍 当前 IP 被 Binance 限制，请切换至 Bybit 或 OKX")
            st.info("👉 在左侧边栏将交易所改为 bybit 或 okx")
        else:
            st.error(f"交易所初始化失败: {error_msg}")
    return None
