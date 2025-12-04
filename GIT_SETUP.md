# Git初期化からプッシュまでの手順

## 前提条件
- GitHubでリポジトリを作成済み（例: https://github.com/your-username/keiba-scenario-ai）
- プロジェクトディレクトリ: keiba-scenario-ai

---

## ステップ1: Gitリポジトリを初期化

```bash
cd keiba-scenario-ai
git init
```

---

## ステップ2: 全ファイルをステージング

```bash
# 全ファイルを追加
git add .

# 追加されたファイルを確認
git status
```

---

## ステップ3: 最初のコミット

```bash
git commit -m "Initial commit: Phase 1 implementation complete

- Timeline manager (walk-forward CV, leak prevention)
- Base model (LightGBM + calibration)
- Scenario adjustment models (Baba, Pace)
- Probability integration (log-odds + Softmax)
- Backtest engine (odds timing, takeout rate)
- SHAP explainer + text generation
- FastAPI + MLflow integration
- Complete documentation"
```

---

## ステップ4: メインブランチ名を設定（必要な場合）

```bash
# デフォルトブランチをmainに変更（Gitの設定による）
git branch -M main
```

---

## ステップ5: リモートリポジトリを追加

**あなたのリポジトリURLに置き換えてください**

```bash
# HTTPSの場合
git remote add origin https://github.com/your-username/keiba-scenario-ai.git

# または SSHの場合
git remote add origin git@github.com:your-username/keiba-scenario-ai.git
```

確認:
```bash
git remote -v
```

---

## ステップ6: プッシュ

```bash
# 初回プッシュ（-u でトラッキング設定）
git push -u origin main
```

---

## 完了！

リポジトリURL: https://github.com/your-username/keiba-scenario-ai

---

## 📋 後続の変更をプッシュする場合

```bash
# ファイルを変更後
git add .
git commit -m "変更内容の説明"
git push
```

---

## 🔧 トラブルシューティング

### エラー: "remote origin already exists"
```bash
git remote remove origin
git remote add origin <your-repo-url>
```

### エラー: "failed to push some refs"
```bash
# リモートの変更を取得してマージ
git pull origin main --rebase
git push origin main
```

### ブランチ名が master の場合
```bash
# mainに変更
git branch -M main
git push -u origin main
```

---

## 📝 .gitignore の内容

以下のファイルは自動的に除外されます：
- `__pycache__/` - Pythonキャッシュ
- `venv/` - 仮想環境
- `data/` - データファイル（大きいため）
- `models/*.pkl` - 学習済みモデル（大きいため）
- `mlruns/` - MLflow実験ログ
- `.env` - 環境変数（秘密情報）

---

## 🎯 プロジェクト情報

**プロジェクト名**: keiba-scenario-ai
**説明**: 競馬予想AIシステム - シナリオ補正型確率予測エンジン
**Phase**: Phase 1（無料データのみ）
**実装完了**: 2024年12月4日

**主要機能**:
- ✅ データリーク防止（TimelineManager）
- ✅ ウォークフォワードCV
- ✅ 確率キャリブレーション（Platt/Isotonic）
- ✅ シナリオ補正（Baba, Pace）
- ✅ log-odds統合 + Softmax正規化
- ✅ バックテストエンジン
- ✅ SHAP説明可能AI
- ✅ FastAPI + MLflow

**技術スタック**:
- Python 3.11+
- LightGBM
- scikit-learn
- FastAPI
- MLflow
- SHAP

---

## 📚 次のステップ

1. GitHub ActionsでCI/CD設定
2. Dockerコンテナ化
3. 本番データでバックテスト
4. Phase 2（JRDBデータ統合）へ
