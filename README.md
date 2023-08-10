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

## MEMO
#### 2023/7/29
とりあえずA4000 SFF Adaで動かす。BSは16が限界。

#### 2023/8/6
promptのテキストを入れると、スコアが安定してこない。要約なので、要約文を入れることでどれくらい要約できているのか？
という部分で貢献しような気がするが、結果としては悪化している。入力しているtextは以下のような形で作成している。
```python
output_df["full_text"] = (
            output_df["prompt_question"]
            + sep
            + output_df["prompt_title"]
            + sep
            + output_df["text"]
            + sep
            + output_df["prompt_text"]
        )
```
full textの与え方が問題なのか？debertaのモデルとしては基本的にはmax_lengthが1532なので、promptが中途半場に入力されてノイズになっているのか？
そうなると、prompt_textの使い方なども重要になっていくる。

またcvの乖離問題も解決していない。基本的にCVよりもLBがよい状態になっている。CVの分割はGroupKFoldで行っている。

このCV戦略が問題ないのか？公開NotebookではMultiStratifiedKFoldを使用していが、連続値では問題があるのがわかっているので
使用しにくい。また、prompt id がtrain, test(public), test(private)で異なるものなので、GroupKFoldがいいのではないかと考えている。

#### 2023/8/7
freeze layerに影響が出るのかを検討する。

| freeze layer |CV|LB|
|--------------|---|---|
| 0(exp020)    |0.535|0.535|
| 2(exp025)    |0.584|0.535|
| 3(exp026)    |0.584|0.535|
| 4(exp027)    |0.584|0.535|
| 5(exp028)    |0.584|0.535|


```
poetry run python src/pipeline.py experiment_name=029 model_name=microsoft/deberta-v3-base freeze_layer=3 split.name=GroupKFold dataset.params.max_len=1532 loss.name=MCRMSELoss
```
