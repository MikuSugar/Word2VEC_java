package com.ansj.vec;

import com.ansj.vec.domain.HiddenNeuron;
import com.ansj.vec.domain.Neuron;
import com.ansj.vec.domain.WordNeuron;
import com.ansj.vec.util.Haffman;
import com.ansj.vec.util.MapCount;

import java.io.*;
import java.nio.file.Files;
import java.util.*;
import java.util.Map.Entry;
import java.util.stream.Collectors;

public class Learn
{

    private final Map<String, Neuron> wordMap = new HashMap<>();

    private MapCount<String> mc;

    /**
     * 训练多少个特征
     */
    private int layerSize = 200;

    /**
     * 上下文窗口大小
     */
    private int window = 5;

    private double sample = 1e-3;

    private double alpha = 0.025;

    private double startingAlpha = alpha;

    public int EXP_TABLE_SIZE = 1000;

    private Boolean isCbow = false;

    private final double[] expTable = new double[EXP_TABLE_SIZE];

    private int trainWordsCount = 0;

    private int MAX_EXP = 6;

    public void setNegative(int negative)
    {
        this.negative = negative;
    }

    private int negative = 0;

    private List<String> wordLists;

    private int[] table;

    private static final int TABLE_SIZE = 100000000;

    private final Random random = new Random();

    public Learn(Boolean isCbow, Integer layerSize, Integer window, Double alpha, Double sample)
    {
        createExpTable();
        if (isCbow != null)
        {
            this.isCbow = isCbow;
        }
        if (layerSize != null)
            this.layerSize = layerSize;
        if (window != null)
            this.window = window;
        if (alpha != null)
            this.alpha = alpha;
        if (sample != null)
            this.sample = sample;
    }

    public Learn()
    {
        createExpTable();
    }

    /**
     * Trains a model using the given file.
     *
     * @param file the file containing the training data
     * @throws IOException if an I/O error occurs while reading the file
     */
    private void trainModel(File file) throws IOException
    {
        Random random = new Random();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(Files.newInputStream(file.toPath()))))
        {
            String line;
            int wordCount = 0;
            int lastWordCount = 0;
            int wordCountActual = 0;
            while ((line = br.readLine()) != null)
            {
                if (wordCount - lastWordCount > 10000)
                {
                    System.out.println("alpha:" + alpha + "\tProgress: " + String.format("%.4f",
                            (wordCountActual / (double)(trainWordsCount)) * 100) + "%");
                    wordCountActual += wordCount - lastWordCount;
                    lastWordCount = wordCount;
                    alpha = startingAlpha * (1 - wordCountActual / (double)(trainWordsCount + 1));
                    if (alpha < startingAlpha * 0.0001)
                    {
                        alpha = startingAlpha * 0.0001;
                    }
                }
                String[] strs = line.split("[\\s　]+");
                wordCount += strs.length;
                List<WordNeuron> sentence = new ArrayList<>(strs.length);
                for (String str : strs)
                {
                    Neuron entry = wordMap.get(str);
                    if (entry == null)
                    {
                        continue;
                    }
                    // The subsampling randomly discards frequent words while
                    // keeping the
                    // ranking same
                    if (sample > 0)
                    {
                        double ran = (Math.sqrt(
                                entry.freq / (sample * trainWordsCount)) + 1) * (sample * trainWordsCount) / entry.freq;

                        if (ran < random.nextDouble())
                        {
                            continue;
                        }
                    }
                    sentence.add((WordNeuron)entry);
                }

                for (int index = 0; index < sentence.size(); index++)
                {

                    if (isCbow)
                    {
                        cbowGram(index, sentence, random.nextInt() % window);
                    }
                    else
                    {
                        skipGram(index, sentence, random.nextInt() % window);
                    }
                }

            }
            System.out.println("Vocab size: " + wordMap.size());
            System.out.println("Words in train file: " + trainWordsCount);
            System.out.println("success train over!");
        }
    }

    private void trainModel(List<int[]> data)
    {
        Random random = new Random();
        int wordCount = 0;
        int lastWordCount = 0;
        int wordCountActual = 0;
        for (int[] line : data)
        {
            if (wordCount - lastWordCount > 10000)
            {
                System.out.println("alpha:" + alpha + "\tProgress: " + String.format("%.4f",
                        (wordCountActual / (double)(trainWordsCount)) * 100) + "%");
                wordCountActual += wordCount - lastWordCount;
                lastWordCount = wordCount;
                alpha = startingAlpha * (1 - wordCountActual / (double)(trainWordsCount + 1));
                if (alpha < startingAlpha * 0.0001)
                {
                    alpha = startingAlpha * 0.0001;
                }
            }
            String[] strs = new String[line.length];
            for (int i = 0; i < line.length; i++)
            {
                strs[i] = String.valueOf(line[i]);
            }

            wordCount += strs.length;
            List<WordNeuron> sentence = new ArrayList<>(strs.length);
            for (String str : strs)
            {
                Neuron entry = wordMap.get(str);
                if (entry == null)
                {
                    continue;
                }
                // The subsampling randomly discards frequent words while
                // keeping the
                // ranking same
                if (sample > 0)
                {
                    double ran = (Math.sqrt(
                            entry.freq / (sample * trainWordsCount)) + 1) * (sample * trainWordsCount) / entry.freq;

                    if (ran < random.nextDouble())
                    {
                        continue;
                    }
                }
                sentence.add((WordNeuron)entry);
            }

            for (int index = 0; index < sentence.size(); index++)
            {

                if (isCbow)
                {
                    cbowGram(index, sentence, random.nextInt() % window);
                }
                else
                {
                    skipGram(index, sentence, random.nextInt() % window);
                }
            }

        }
        System.out.println("Vocab size: " + wordMap.size());
        System.out.println("Words in train file: " + trainWordsCount);
        System.out.println("success train over!");
    }

    /**
     * skipGram
     *
     * @param index    the index of the word in the sentence
     * @param sentence the sentence containing the word
     * @param b        the context window size
     */
    private void skipGram(int index, List<WordNeuron> sentence, int b)
    {
        WordNeuron word = sentence.get(index);
        int a, c;
        for (a = b; a < window * 2 + 1 - b; a++)
        {
            if (a == window)
            {
                continue;
            }
            c = index - window + a;
            if (c < 0 || c >= sentence.size())
            {
                continue;
            }

            double[] neu1e = new double[layerSize];// 误差项
            WordNeuron we = sentence.get(c);
            // HIERARCHICAL SOFTMAX
            if (negative <= 0)
            {
                skipGramSoftMax(word, we, neu1e);

            }
            // NEGATIVE SAMPLING
            else
            {
                skipGramNegativeSampling(we, neu1e);
            }
            // Learn weights input -> hidden
            for (int j = 0; j < layerSize; j++)
            {
                we.syn0[j] += neu1e[j];
            }
        }

    }

    private void skipGramNegativeSampling(WordNeuron cur, double[] neu1e)
    {
        String target;
        int label;
        for (int d = 0; d < negative + 1; d++)
        {
            if (d == 0)
            {
                target = cur.name;
                label = 1;
            }
            else
            {
                target = wordLists.get(table[random.nextInt(table.length)]);
                if (target.equals(cur.name))
                {
                    continue;
                }
                label = 0;
            }
            final WordNeuron targetNeuron = (WordNeuron)wordMap.get(target);
            final double g = getG(cur, targetNeuron, label);
            for (int j = 0; j < layerSize; j++)
            {
                neu1e[j] += g * targetNeuron.negativeSyn1[j];
                targetNeuron.negativeSyn1[j] += g * cur.syn0[j];
            }
        }
    }

    private void skipGramSoftMax(WordNeuron word, WordNeuron we, double[] neu1e)
    {
        int c;
        List<Neuron> neurons = word.neurons;
        for (int i = 0; i < neurons.size(); i++)
        {
            HiddenNeuron out = (HiddenNeuron)neurons.get(i);
            // Propagate hidden -> output
            double f = dot(we, out.syn1);
            if (f <= -MAX_EXP || f >= MAX_EXP)
            {
                continue;
            }
            else
            {
                //这行代码的作用是将预测值 f 进行映射，将其从范围 [-MAX_EXP, MAX_EXP] 映射到 [0, EXP_TABLE_SIZE]
                f = (f + MAX_EXP) * ((double)EXP_TABLE_SIZE / MAX_EXP / 2);
                f = expTable[(int)f];
            }
            // 'g' is the gradient multiplied by the learning rate
            double g = (1 - word.codeArr[i] - f) * alpha;
            // Propagate errors output -> hidden
            for (c = 0; c < layerSize; c++)
            {
                //Propagate errors output -> hidden
                neu1e[c] += g * out.syn1[c];
                //Learn weights hidden -> output
                out.syn1[c] += g * we.syn0[c];
            }
        }
    }

    private double getG(WordNeuron we, WordNeuron targetNeuron, int label)
    {
        double f = dot(we, targetNeuron.negativeSyn1);
        double g;
        if (f > MAX_EXP)
        {
            g = (label - 1) * alpha;
        }
        else if (f < -MAX_EXP)
        {
            g = label * alpha;
        }
        else
        {
            g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        }
        return g;
    }

    private double dot(WordNeuron we, double[] targetNeuron)
    {
        double f = 0d;
        for (int j = 0; j < layerSize; j++)
        {
            f += we.syn0[j] * targetNeuron[j];
        }
        return f;
    }

    /**
     * cbowGram
     *
     * @param index    the index of the word in the sentence
     * @param sentence the sentence containing the word
     * @param b        the context window size
     */
    private void cbowGram(int index, List<WordNeuron> sentence, int b)
    {
        WordNeuron word = sentence.get(index);
        int a, c;

        List<Neuron> neurons = word.neurons;
        double[] neu1e = new double[layerSize];// 误差项
        double[] neu1 = new double[layerSize];// 误差项
        WordNeuron last_word;

        for (a = b; a < window * 2 + 1 - b; a++)
            if (a != window)
            {
                c = index - window + a;
                if (c < 0)
                    continue;
                if (c >= sentence.size())
                    continue;
                last_word = sentence.get(c);
                if (last_word == null)
                    continue;
                for (c = 0; c < layerSize; c++)
                    neu1[c] += last_word.syn0[c];
            }

        // HIERARCHICAL SOFTMAX
        for (int d = 0; d < neurons.size(); d++)
        {
            HiddenNeuron out = (HiddenNeuron)neurons.get(d);
            double f = 0;
            // Propagate hidden -> output
            for (c = 0; c < layerSize; c++)
                f += neu1[c] * out.syn1[c];
            if (f <= -MAX_EXP)
                continue;
            else if (f >= MAX_EXP)
                continue;
            else
                f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
            // 'g' is the gradient multiplied by the learning rate
            // double g = (1 - word.codeArr[d] - f) * alpha;
            // double g = f*(1-f)*( word.codeArr[i] - f) * alpha;
            double g = f * (1 - f) * (word.codeArr[d] - f) * alpha;
            //
            for (c = 0; c < layerSize; c++)
            {
                neu1e[c] += g * out.syn1[c];
            }
            // Learn weights hidden -> output
            for (c = 0; c < layerSize; c++)
            {
                out.syn1[c] += g * neu1[c];
            }
        }
        for (a = b; a < window * 2 + 1 - b; a++)
        {
            if (a != window)
            {
                c = index - window + a;
                if (c < 0)
                    continue;
                if (c >= sentence.size())
                    continue;
                last_word = sentence.get(c);
                if (last_word == null)
                    continue;
                for (c = 0; c < layerSize; c++)
                    last_word.syn0[c] += neu1e[c];
            }

        }
    }

    /**
     * Read vocabulary from a given file and count word frequencies.
     *
     * @param file the file to read
     * @throws IOException if an I/O error occurs while reading the file
     */
    private void readVocab(File file) throws IOException
    {
        this.mc = new MapCount<>();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(Files.newInputStream(file.toPath()))))
        {
            String temp;
            while ((temp = br.readLine()) != null)
            {
                String[] split = temp.split("[\\s　]+");// 修改，支持出现多个半角全角空格，制表符分隔。
                trainWordsCount += split.length;
                for (String string : split)
                {
                    mc.add(string);
                }
            }
        }
        helpRead();
    }

    private void helpRead()
    {
        for (Entry<String, Integer> element : mc.get().entrySet())
        {
            wordMap.put(element.getKey(),
                    new WordNeuron(element.getKey(), (double)element.getValue() / mc.size(), layerSize));
        }
        if (negative > 0)
        {
            wordLists = new ArrayList<>(mc.get().keySet());
        }
        else
        {
            this.mc = null;
        }
    }

    private void readVocab(List<int[]> data)
    {
        this.mc = new MapCount<>();
        for (int[] line : data)
        {
            trainWordsCount += line.length;
            for (int v : line)
            {
                mc.add(String.valueOf(v));
            }
        }
        helpRead();
    }

    /**
     * 对文本进行预分类
     *
     * @param files
     * @throws IOException
     * @throws FileNotFoundException
     */
    private void readVocabWithSupervised(File[] files) throws IOException
    {
        for (int category = 0; category < files.length; category++)
        {
            // 对多个文件学习
            MapCount<String> mc = new MapCount<>();
            try (
                    BufferedReader br = new BufferedReader(
                            new InputStreamReader(Files.newInputStream(files[category].toPath())))
            )
            {
                String temp = null;
                while ((temp = br.readLine()) != null)
                {
                    String[] split = temp.split("[\\s　]+");
                    trainWordsCount += split.length;
                    for (String string : split)
                    {
                        mc.add(string);
                    }
                }
            }
            for (Entry<String, Integer> element : mc.get().entrySet())
            {
                double tarFreq = (double)element.getValue() / mc.size();
                if (wordMap.get(element.getKey()) != null)
                {
                    double srcFreq = wordMap.get(element.getKey()).freq;
                    if (srcFreq >= tarFreq)
                    {
                        continue;
                    }
                    else
                    {
                        Neuron wordNeuron = wordMap.get(element.getKey());
                        wordNeuron.category = category;
                        wordNeuron.freq = tarFreq;
                    }
                }
                else
                {
                    wordMap.put(element.getKey(), new WordNeuron(element.getKey(), tarFreq, category, layerSize));
                }
            }
        }
    }

    /**
     * Precompute the exp() table f(x) = x / (x + 1)
     */
    private void createExpTable()
    {
        for (int i = 0; i < EXP_TABLE_SIZE; i++)
        {
            expTable[i] = Math.exp(((i / (double)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP));
            expTable[i] = expTable[i] / (expTable[i] + 1);
        }
    }

    /**
     * Learn from a file
     *
     * @param file The file to learn from
     * @throws IOException If an I/O error occurs
     */
    public void learnFile(File file) throws IOException
    {
        readVocab(file);
        if (negative > 0)
        {
            initNegative();
        }
        new Haffman(layerSize).make(wordMap.values());

        // 查找每个神经元
        for (Neuron neuron : wordMap.values())
        {
            ((WordNeuron)neuron).makeNeurons();
        }

        trainModel(file);
    }

    public void learnData(List<int[]> data)
    {
        readVocab(data);
        if (negative > 0)
        {
            initNegative();
        }
        new Haffman(layerSize).make(wordMap.values());
        for (Neuron neuron : wordMap.values())
        {
            ((WordNeuron)neuron).makeNeurons();
        }
        trainModel(data);
    }

    /**
     * Initialize the negative sampling table
     */
    private void initNegative()
    {
        this.table = new int[TABLE_SIZE];
        Arrays.fill(table, -1);
        final HashMap<String, Integer> mapCount = mc.get();
        long trainWordsPow = 0;
        final double power = 0.75;
        for (int count : mapCount.values())
        {
            trainWordsPow += (long)Math.pow(count, power);
        }
        int i = 0;
        double d1 = Math.pow(mapCount.get(wordLists.get(0)), power) / trainWordsPow;
        for (int a = 0; a < TABLE_SIZE; a++)
        {
            table[a] = i;
            if ((double)a / TABLE_SIZE > d1)
            {
                i++;
                d1 += Math.pow(mapCount.get(wordLists.get(i)), power) / trainWordsPow;
            }
            if (i >= wordLists.size())
            {
                i = wordLists.size() - 1;
            }
        }
        for (Neuron value : wordMap.values())
        {
            WordNeuron wordNeuron = (WordNeuron)value;
            wordNeuron.negativeSyn1 = new double[layerSize];
        }

    }

    /**
     * 根据预分类的文件学习
     *
     * @param summaryFile     合并文件
     * @param classifiedFiles 分类文件
     */
    public void learnFile(File summaryFile, File[] classifiedFiles) throws IOException
    {
        readVocabWithSupervised(classifiedFiles);
        new Haffman(layerSize).make(wordMap.values());
        // 查找每个神经元
        for (Neuron neuron : wordMap.values())
        {
            ((WordNeuron)neuron).makeNeurons();
        }
        trainModel(summaryFile);
    }

    /**
     * 保存模型
     */
    public void saveModel(File file) throws IOException
    {
        try (
                DataOutputStream dataOutputStream = new DataOutputStream(
                        new BufferedOutputStream(Files.newOutputStream(file.toPath())))
        )
        {
            dataOutputStream.writeInt(wordMap.size());
            dataOutputStream.writeInt(layerSize);
            double[] syn0;
            for (Entry<String, Neuron> element : wordMap.entrySet())
            {
                dataOutputStream.writeUTF(element.getKey());
                syn0 = ((WordNeuron)element.getValue()).syn0;
                for (double d : syn0)
                {
                    dataOutputStream.writeDouble(d);
                }
            }
        }
    }

    public int getLayerSize()
    {
        return layerSize;
    }

    public void setLayerSize(int layerSize)
    {
        this.layerSize = layerSize;
    }

    public int getWindow()
    {
        return window;
    }

    public void setWindow(int window)
    {
        this.window = window;
    }

    public double getSample()
    {
        return sample;
    }

    public void setSample(double sample)
    {
        this.sample = sample;
    }

    public double getAlpha()
    {
        return alpha;
    }

    public void setAlpha(double alpha)
    {
        this.alpha = alpha;
        this.startingAlpha = alpha;
    }

    public Boolean getIsCbow()
    {
        return isCbow;
    }

    public void setIsCbow(Boolean isCbow)
    {
        this.isCbow = isCbow;
    }

    public void setMAX_EXP(int MAX_EXP)
    {
        this.MAX_EXP = MAX_EXP;
    }

    public Map<String, Neuron> getWordMap()
    {
        return wordMap;
    }
}
