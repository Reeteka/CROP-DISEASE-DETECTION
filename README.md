# CROP-DISEASE-DETECTION
BLOCKCHAIN -ENABLED DEEP LEARNING FOR PRECISION CROP DISEASE DETECTION

# 📦 Crop Disease Detection with Blockchain Feedback

A mini-project that combines deep learning and blockchain to enable precise crop disease detection and immutable feedback logging.

---

## 🔍 Project Overview

This project uses a YOLOv8-based deep learning model to detect crop diseases from images. Once a prediction is made, the feedback is logged onto the Ethereum blockchain via a smart contract to ensure tamper-proof, permanent storage.

---

## 🛠️ Tech Stack

- **Python** (YOLOv8 for image detection)
- **Solidity** (Smart contract for Ethereum)
- **Google Colab** (for model training and testing)
- **MetaMask + Alchemy** (for Ethereum transactions)
- **CSV** (Local feedback log)
- **Ethereum testnet** (for smart contract deployment)

---

## 📁 Project Structure

```
code crop/
├── CropDisease.sol                      # Smart contract code
├── yolocode.py                          # Python script for prediction
├── predictions_log1 feedback logged.csv  # Logged feedback from users
├── projectreport-Reetu.pdf              # Project documentation
└── desktop.ini                          # (Ignore this)
```

---

## ⚙️ How to Run

1. **Set up Python environment**
   - Install dependencies (e.g., ultralytics, web3)

   ```bash
   pip install ultralytics web3
   ```

2. **Run YOLOv8 Detection**

   ```bash
   python yolocode.py
   ```

3. **Deploy Smart Contract**
   - Use Remix IDE or Hardhat to deploy `CropDisease.sol` on the Ethereum testnet.

4. **Connect with MetaMask**
   - Connect your wallet using MetaMask
   - Interact with the smart contract to log feedback

---

## ✅ Features

- 🔍 Fast and accurate disease detection using YOLOv8
- 🔗 Blockchain-backed logging of user feedback
- 🛡️ Tamper-proof storage on Ethereum

---

## 📄 License

This project is for academic and personal use only. Please contact the author for any commercial use.
