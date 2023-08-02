# TODO
在这个项目基础上实现node2vec👷
# README

源项目链接：https://github.com/NLPchina/Word2VEC_java

在源项目中做了如下处理：

1.确保语料文本文件是UTF-8编码，附带了语料corpus.txt，训练模型文件model.bin因太大(120M)没有提交，需要自己本地训练(LearnTest.class)，训练时间大概几十分钟。

2.源作者提供的语料是用制表符切割的词组，但是代码是根据空格切割，需要将制表符全部替换成空格。或者修改代码：Learn.java 271行，修改成String[] split = temp.split("[\s　]+");支持同时出现多个半角或全角空格，或制表符分隔。

3.发现一个bug
Word2Vec中2个distance方法中，min = result.last().score; 应该放在resultSize < result.size()块里。
只有当结果数已经大于resultSize，才能将最后一个得分数赋予min，作为以后最小允许得分。结果数不大于resultSize不能赋予给min。

运行Word2VecTest.class，距离最近词，计算词距离，聚类等：

    public static void main(String[] args) throws IOException {
        Word2vec vec = new Word2vec();
        vec.loadJavaModel("model.bin");
        // 距离最近的词
        System.out.println(vec.distance("邓小平"));
        System.out.println(vec.distance("魔术队"));
        System.out.println(vec.distance("过年"));
        System.out.println(vec.distance(Arrays.asList("香港", "澳门")));
        // // 计算词之间的距离
        HashMap<String, float[]> map = vec.getWordMap();
        float[] center1 = map.get("春节");
        float[] center2 = map.get("过年");
        double dics = 0;
        for (int i = 0; i < center1.length; i++) {
            dics += center1[i] * center2[i];
        }
        System.out.println(dics);
        // 距离计算
        System.out.println(vec.analogy("毛泽东", "邓小平", "毛泽东思想"));
        System.out.println(vec.analogy("女人", "男人", "女王"));
        System.out.println(vec.analogy("北京", "中国", "巴黎"));
        // 聚类
        WordKmeans wordKmeans = new WordKmeans(vec.getWordMap(), 50, 50);
        Classes[] explain = wordKmeans.explain();
        for (int i = 0; i < explain.length; i++) {
            System.out.println("--------" + i + "---------");
            System.out.println(explain[i].getTop(10));
        }
    }
