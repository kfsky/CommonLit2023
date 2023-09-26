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

- これでどこまで精度向上が図れるか？
結果として、stopwordを入れないで精度の向上がおきた。おそらく、必要な情報のみをprompt側に残すことで、要約ができているかをモデルが認識しやすくなったのではないかと考えている。
- あと、ピリオドを除外している。一つの大きな文章（文字列）として扱う工夫をしている。
```
With one member trimming beef in a cannery and another working in a sausage factory the family had a first hand knowledge of the great majority of Packingtown swindles For it was the custom as they found whenever meat was so spoiled that it could not be used for anything else either to can it or else to chop it up into 
sausage With what had been told them by Jonas who had worked in the pickle rooms they could now study the whole of the spoiled meat industry on the inside and read a new and grim meaning into that old Packingtown jest that they use everything of the pig except the squeal Jonas had told them how the meat that 
was taken out of pickle would often be found sour and how they would rub it up with soda to take away the smell and sell it to be eaten on free lunch counters also of all the miracles of chemistry which they performed giving to any sort of meat fresh or salted whole or chopped any color and any flavor and 
any odor they chose In the pickling of hams they had an ingenious apparatus by which they saved time and increased the capacity of the plant a machine consisting of a hollow needle attached to a pump by plunging this needle into the meat and working with his foot a man could fill a ham with pickle in a few 
seconds And yet in spite of this there would be hams found spoiled some of them with an odor so bad that a man could hardly bear to be in the room with them To pump into these the packers had a second and much stronger pickle which destroyed the odor a process known to the workers as giving them thirty percent
Also after the hams had been smoked there would be found some that had gone to the bad Formerly these had been sold as Number Three Grade but later on some ingenious person had hit upon a new device and now they would extract the bone about which the bad part generally lay and insert in the hole a white 
hot iron After this invention there was no longer Number One Two and Three Grade there was only Number One Grade The packers were always originating such schemes they had what they called boneless hams which were all the odds and ends of pork stuffed into casings and California hams which were the shoulders 
with big knuckle joints and nearly all the meat cut out and fancy skinned hams which were made of the oldest hogs whose skins were so heavy and coarse that no one would buy them that is until they had been cooked and chopped fine and labeled head cheese It was only when the whole ham was spoiled that it 
came into the department of Elzbieta Cut up by the two thousand revolutions a minute flyers and mixed with half a ton of other meat no odor that ever was in a ham could make any difference There was never the least attention paid to what was cut up for sausage there would come all the way back from Europe 
old sausage that had been rejected and that was moldy and white it would be dosed with borax and glycerin and dumped into the hoppers and made over again for home consumption There would be meat that had tumbled out on the floor in the dirt and sawdust where the workers had tramped and spit uncounted billions 
of consumption germs There would be meat stored in great piles in rooms and the water from leaky roofs would drip over it and thousands of rats would race about on it It was too dark in these storage places to see well but a man could run his hand over these piles of meat and sweep off handfuls of the dried 
dung of rats These rats were nuisances and the packers would put poisoned bread out for them they would die and then rats bread and meat would go into the hoppers together This is no fairy story and no joke the meat would be shoveled into carts and the man who did the shoveling would not trouble to lift 
out a rat even when he saw one there were things that went into the sausage in comparison with which a poisoned rat was a tidbit There was no place for the men to wash their hands before they ate their dinner and so they made a practice of washing them in the water that was to be ladled into the sausage 
There were the butt ends of smoked meat and the scraps of corned beef and all the odds and ends of the waste of the plants that would be dumped into old barrels in the cellar and left there Under the system of rigid economy which the packers enforced there were some jobs that it only paid to do once in a long 
time and among these was the cleaning out of the waste barrels Every spring they did it and in the barrels would be dirt and rust and old nails and stale water and cartload after cartload of it would be taken up and dumped into the hoppers with fresh meat and sent out to the public is breakfast Some of it 
they would make into smoked sausage but as the smoking took time and was therefore expensive they would call upon their chemistry department and preserve it with borax and color it with gelatin to make it brown All of their sausage came out of the same bowl but when they came to wrap it they would stamp 
some of it special and for this they would charge two cents more a pound
```


