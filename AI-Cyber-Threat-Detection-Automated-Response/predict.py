import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load model & encoder
xgb_model = joblib.load("xgb_cyber_model.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# Define categorical mappings if needed
proto_map = {"tcp":0, "udp":1, "icmp":2}
service_map = {"http":0, "ftp":1, "smtp":2}
state_map = {"FIN":0, "CON":1, "INT":2}

# This is the exact list of columns your model was trained on
model_columns = [
    'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes',
    'rate', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
    'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean',
    'trans_depth', 'response_body_len', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
    'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'is_sm_ips_ports'
]

def preprocess_record(record):
    df = pd.DataFrame([record])
    
    # Map categorical features
    df['proto'] = df['proto'].map(proto_map)
    df['service'] = df['service'].map(service_map)
    df['state'] = df['state'].map(state_map)
    
    # Ensure all columns exist in same order as training
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0  # fill missing numeric/categorical features with 0
    df = df[model_columns]
    
    return df

def predict_attack(record):
    df = preprocess_record(record)
    pred = xgb_model.predict(df)[0]
    prob = xgb_model.predict_proba(df).max()
    attack_label = target_encoder.inverse_transform([pred])[0]

    if attack_label in ["DoS", "Exploits", "Backdoor", "Shellcode", "Worms"]:
        response = "⚠️ ALERT: Suspicious activity detected! Take immediate action!"
    else:
        response = "✅ Normal activity."

    return attack_label, prob, response

if __name__ == "__main__":
    # Example new record (fill missing features as 0 if unknown)
    new_record = {
        'dur':0, 'proto':'tcp', 'service':'http', 'state':'FIN', 'spkts':10, 'dpkts':5,
        'sbytes':181, 'dbytes':5450, 'rate':0.0, 'sload':0.0, 'dload':0.0, 'sloss':0,
        'dloss':0, 'sinpkt':0, 'dinpkt':0, 'sjit':0, 'djit':0, 'swin':0, 'stcpb':0,
        'dtcpb':0, 'dwin':0, 'tcprtt':0, 'synack':0, 'ackdat':0, 'smean':0.0, 'dmean':0.0,
        'trans_depth':0, 'response_body_len':0, 'ct_src_dport_ltm':0, 'ct_dst_sport_ltm':0,
        'is_ftp_login':0, 'ct_ftp_cmd':0, 'ct_flw_http_mthd':0, 'is_sm_ips_ports':0
    }

    attack, prob, action = predict_attack(new_record)
    print(f"Predicted Attack Category: {attack}")
    print(f"Prediction Probability: {prob:.2f}")
    print(f"Automated Response: {action}")
