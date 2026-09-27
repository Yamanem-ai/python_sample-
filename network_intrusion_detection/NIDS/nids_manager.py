
import HEX_collection
import NIDS_Transformer


def main():

    print("Loading NIDS Transformer...")

    # NIDSモデルは最初に1回だけロード
    models = NIDS_Transformer.load_model()

    print("NIDS Transformer loaded.")
    print("Start packet capture...")

    # HEX_collectionからpacket_hexを順番に受け取る
    for packet_hex in HEX_collection.capture_packets("en0"):

        print()
        print("Packet received")
        print(f"HEX size: {len(packet_hex)} characters")

        try:

            # NIDS推論
            tags = NIDS_Transformer.predict_packet_top_tags(
                models,
                packet_hex,
                top_k=5,
            )

            print("--- NIDS result ---")

            for tag, probability in tags:
                print(f"{tag}: {probability:.4f}")

            print("-------------------")

        except Exception as e:

            print(f"NIDS inference error: {e}")


if __name__ == "__main__":
    main()


