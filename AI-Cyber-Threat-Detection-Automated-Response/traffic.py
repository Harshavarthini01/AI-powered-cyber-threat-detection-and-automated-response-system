from scapy.all import sniff, IP, TCP
import pandas as pd
import os
print(os.getcwd())
# This will store captured packet info
data = []

# Function called for each packet
def process_packet(pkt):
    if IP in pkt:
        data.append({
            "src": pkt[IP].src,
            "dst": pkt[IP].dst,
            "proto": pkt[IP].proto,
            "len": len(pkt)
        })

# Capture packets for 10 seconds
sniff(prn=process_packet, timeout=10)

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("traffic.csv", index=False)

print("Traffic captured and saved to traffic.csv")
