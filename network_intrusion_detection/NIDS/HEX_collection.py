from scapy.all import sniff, IP, TCP, Raw


def capture_packets(interface="en0"):
    """
    IPv4 + TCP + Payload のパケットだけを取得して、
    Ethernetフレーム全体をHEX文字列として1パケットずつyieldする。
    """

    while True:
        packet_list = sniff(
            iface=interface,
            count=1,
            store=True,
        )

        if not packet_list:
            continue

        packet = packet_list[0]

        # ========================================
        # IPv4 + TCP + Payload のパケットだけを選択
        # ========================================

        if not packet.haslayer(IP):
            continue

        if not packet.haslayer(TCP):
            continue

        if not packet.haslayer(Raw):
            continue

        # ========================================
        # ここまで来たものだけNIDSに渡す
        # ========================================

        packet_hex = bytes(packet).hex()

        yield packet_hex


if __name__ == "__main__":
    print("Start packet capture...")

    for packet_hex in capture_packets("en0"):
        print("Packet received")
        print(f"HEX size: {len(packet_hex)} characters")
        print(packet_hex)
        print("-" * 60)
