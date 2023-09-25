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

## 目的変数の算出方法
### Content
- How well did the summary capture the main idea of the source.(要約は出典の主旨をどの程度捉えていたか。)

- How accurately did the summary capture the details from the source.(要約は出典の詳細をどの程度正確に捉えていたか。)

- How well did the summary transition from one idea to the next.(要約は1つのアイデアから次のアイデアにどれだけうまく移行したか。)

### Wording
- Was the summary written using objective language.(要約は客観的な言葉を使って書かれていたか。)

- Is the summary appropriately paraphrased.(要約は適切に言い換えられているか)

- How well did the summary use texts and syntax.(要約は文章と構文をどの程度うまく使っているか。)


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

| freeze layer | CV    | LB    |
|--------------|-------|-------|
| 0(exp020)    | 0.557 | 0.491 |
| 2(exp025)    | 0.584 | 0.506 |
| 3(exp026)    | 0.563 | 0.496 |
| 4(exp027)    | 0.583 | 0.506 |
| 5(exp028)    | 0.582 | 0.497 |

結果としてはfreeze layer=3で最もLBは良かったけど、seedごとのCVの分布が大きいので、freezeしないほうがいいかもしれない。

#### 2023/8/12
seedごとのばらつきが発生するので、seed値を複数使って学習。0.01くらいは変動してしまうので、3seedの平均で考える必要がありそう。
inferenceのほうもseed averageで行うほうがいいのかはわからない。

typoを修正するとスコアが悪化する。typoは特に気にしないで採点しているのか？

-> なんでなのかよくわかっていない。

各foldでのスコアのばらつきが大きい（目的変数の分布が異なっている）ので、publicLBがどのfoldの目的変数の分布に似ているのかを確認してみる。
exp018(deberta-v3-large)で結果を確認

| fold | CV    | LB    |
|--------------|-------|-------|
| 0    | 0.504 | 0.497 |
| 1    | 0.651 | 0.504 |
| 2    | 0.499 | 0.514 |
| 3    | 0.592 | 0.504 |
| 0, 2    | - | 0.488 |
| total    | 0.562 | 0.488 |

fold1の分布に近い模様。テストデータは17,000であり、その18%がpublicLBなので、データ件数としては3060件。このデータがfold0に近い分布の模様。
ただし、残りのデータが同じ分布との限らない気がするので、各foldの平均をとるのはいいのかも？（fold3のデータが極端なので、そのデータは除外してみるのもいいかもしれない）

#### 2023/8/13
sentence の与え方について検討してなかったので、もう少し考えてみる。textから入力することを考えていなかったので、それで改善するのか確認する。

#### 2023/8/29
stackingが行うと精度が上がることは確認できている。
スタッキングによるbertで取り切れない特徴量を追加することで対応することが可能なのか？

#### 2023/9/15
特殊トークンを追加することで精度向上ができないかを検証している。実施しているのは以下
- 参照している部分を[PARAGRAPH]というトークンに置き換える
- 著者名が記載している部分の単語を[AUTHOR]というトークンに置き換える
- スペルミスの単語を[MISSPELL]というトークンに置き換える

#### 2023/9/25
恐らくシングルモデルで上位にいる方は、要約前の文章を利用している模様。ただし、現状ではCVもLBも下がる状態。使用するためのアイディアとしては以下
- text cleaningをがっつり行う（stopwordの除外も含め）-> 想定の効果としては、必要な情報のみを残すことで要約部分ができているかの精度向上ができないか？

これでどこまで精度向上が図れるか？
