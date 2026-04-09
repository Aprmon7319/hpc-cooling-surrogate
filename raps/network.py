TX_MAX = 10000
RX_MAX = 20000

def network_utilization(tx, rx):
    """Compute average network utilization"""
    tx_util = min(tx / TX_MAX, 1.0)  # Clamp to 1.0
    rx_util = min(rx / RX_MAX, 1.0)
    return (tx_util + rx_util) / 2.0
