import scapy.all as scapy
import binascii

packets = scapy.rdpcap("test.pcap")


def cut(obj, sec):
    result = [obj[i : i + sec] for i in range(0, len(obj), sec)]
    try:
        remanent_count = len(result[0]) % 4
    except Exception as e:
        remanent_count = 0
        print("cut datagram error!")
    if remanent_count == 0:
        pass
    else:
        result = [
            obj[i : i + sec + remanent_count]
            for i in range(0, len(obj), sec + remanent_count)
        ]
    return result


def bigram_generation(packet_datagram, packet_len=64, flag=True):
    result = ""
    generated_datagram = cut(packet_datagram, 1)
    token_count = 0
    for sub_string_index in range(len(generated_datagram)):
        if sub_string_index != (len(generated_datagram) - 1):
            token_count += 1
            if token_count > packet_len:
                break
            else:
                merge_word_bigram = (
                    generated_datagram[sub_string_index]
                    + generated_datagram[sub_string_index + 1]
                )
        else:
            break
        result += merge_word_bigram
        result += " "

    return result


packet_index = 0
packet_count = 0
flow_data_string = ""
for packet in packets:
    packet_count += 1

    packet_data = packet.copy()
    data = binascii.hexlify(bytes(packet_data))
    packet_string = data.decode()[76:]
    a = bigram_generation(packet_string, packet_len=128, flag=True)
    print(a)
    flow_data_string += a
    # break

# print(flow_data_string)
