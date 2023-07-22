# CommonLit2023
https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries

## コンペ概要
このコンテストの目標は、3年生から12年生の生徒が書いた要約の質を評価することです。生徒が原文の主旨と詳細をどれだけ適切に表現しているか、
また要約で使用される言葉の明瞭さ、正確さ、流暢さを評価するモデルを構築します。
モデルをトレーニングするために、実際の生徒の要約のコレクションにアクセスできます。
あなたの作成した要約は、教師が生徒の学習の質を評価するのに役立つだけでなく、学習プラットフォームが生徒に即座にフィードバックを提供するのにも役立ちます。

## データ
このデータセットは、3年生から12年生までの生徒が、さまざまなトピックやジャンルの文章について書いた約24,000の要約で構成されている。これらの要約には、
内容と言い回しの両方についてスコアが割り当てられている。このコンペティションの目的は、未知のトピックに関する要約の内容と語句の得点を予測することである。

## Pipeline
以下のように実行すればよい
```commandline
~/Desktop/CommonLit2023$ poetry run python src/pipeline.py
```

実験ごとに行う場合は、コマンドライン上で設定値を修正すること（実験名は必ず変えること！それ以外のパラメータも変更できる！）
```commandline
~/Desktop/CommonLit2023$ poetry run python src/pipeline.py experiment_name=001 globals.debug=True model_name=microsoft/deberta-v3-small
```

