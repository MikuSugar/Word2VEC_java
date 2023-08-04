package com.ansj.vec;

import com.ansj.vec.domain.WordEntry;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.Map.Entry;

public class Word2vec
{

    private final HashMap<String, double[]> wordMap = new HashMap<>();

    private int words;

    private int size;

    private int topNSize = 40;

    /**
     * 加载模型
     *
     * @param path 模型的路径
     * @throws IOException IOException
     */
    public void loadGoogleModel(String path) throws IOException
    {
        double len;
        double vector;
        try (
                BufferedInputStream bis = new BufferedInputStream(Files.newInputStream(Paths.get(path)));
                DataInputStream dis = new DataInputStream(bis)
        )
        {
            // //读取词数
            words = Integer.parseInt(readString(dis));
            // //大小
            size = Integer.parseInt(readString(dis));
            String word;
            double[] vectors = null;
            for (int i = 0; i < words; i++)
            {
                word = readString(dis);
                vectors = new double[size];
                len = 0;
                for (int j = 0; j < size; j++)
                {
                    vector = readDouble(dis);
                    len += vector * vector;
                    vectors[j] = vector;
                }
                len = Math.sqrt(len);
                for (int j = 0; j < size; j++)
                {
                    vectors[j] /= len;
                }
                wordMap.put(word, vectors);
                dis.read();
            }
        }
    }

    /**
     * 加载模型
     *
     * @param path 模型的路径
     * @throws IOException IOException
     */
    public void loadJavaModel(String path) throws IOException
    {
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(Files.newInputStream(Paths.get(path)))))
        {
            words = dis.readInt();
            size = dis.readInt();

            double vector;

            String key;
            double[] value;
            for (int i = 0; i < words; i++)
            {
                double len = 0;
                key = dis.readUTF();
                value = new double[size];
                for (int j = 0; j < size; j++)
                {
                    vector = dis.readDouble();
                    len += vector * vector;
                    value[j] = vector;
                }

                len = Math.sqrt(len);

                for (int j = 0; j < size; j++)
                {
                    value[j] /= len;
                }
                wordMap.put(key, value);
            }

        }
    }

    private static final int MAX_SIZE = 50;

    /**
     * 近义词推断
     *
     * @param word0 第一个单词
     * @param word1 第二个单词
     * @param word2 第三个单词
     * @return 与给定单词相关联的按相似度排序的单词列表
     */
    public TreeSet<WordEntry> analogy(String word0, String word1, String word2)
    {
        double[] wv0 = getWordVector(word0);
        double[] wv1 = getWordVector(word1);
        double[] wv2 = getWordVector(word2);

        if (wv1 == null || wv2 == null || wv0 == null)
        {
            return null;
        }
        double[] wordVector = new double[size];
        for (int i = 0; i < size; i++)
        {
            wordVector[i] = wv1[i] - wv0[i] + wv2[i];
        }
        double[] tempVector;
        String name;
        List<WordEntry> topNEntries = new ArrayList<>(topNSize);
        for (Entry<String, double[]> entry : wordMap.entrySet())
        {
            name = entry.getKey();
            if (name.equals(word0) || name.equals(word1) || name.equals(word2))
            {
                continue;
            }
            double dist = 0;
            tempVector = entry.getValue();
            for (int i = 0; i < wordVector.length; i++)
            {
                dist += wordVector[i] * tempVector[i];
            }
            insertTopN(name, dist, topNEntries);
        }
        return new TreeSet<>(topNEntries);
    }

    private void insertTopN(String name, double score, List<WordEntry> wordsEntry)
    {
        if (wordsEntry.size() < topNSize)
        {
            wordsEntry.add(new WordEntry(name, score));
            return;
        }
        double min = Double.MAX_VALUE;
        int minOffe = 0;
        for (int i = 0; i < topNSize; i++)
        {
            WordEntry wordEntry = wordsEntry.get(i);
            if (min > wordEntry.score)
            {
                min = wordEntry.score;
                minOffe = i;
            }
        }
        if (score > min)
        {
            wordsEntry.set(minOffe, new WordEntry(name, score));
        }
    }

    public Set<WordEntry> distance(String queryWord)
    {
        double[] center = wordMap.get(queryWord);
        if (center == null)
        {
            return Collections.emptySet();
        }
        int resultSize = Math.min(wordMap.size(), topNSize);
        TreeSet<WordEntry> result = new TreeSet<>();
        double min = Double.MIN_VALUE;
        for (Map.Entry<String, double[]> entry : wordMap.entrySet())
        {
            String key = entry.getKey();
            if (key == null || key.equals(queryWord))
            {
                continue;
            }
            double[] vector = entry.getValue();
            double dist = 0;
            for (int i = 0; i < vector.length; i++)
            {
                dist += center[i] * vector[i];
            }
            if (dist <= min)
            {
                continue;
            }
            result.add(new WordEntry(entry.getKey(), dist));
            if (result.size() > resultSize)
            {
                result.pollLast();
                if (!result.isEmpty())
                {
                    min = result.last().score;
                }
            }
        }
        return result;
    }

    public Set<WordEntry> distance(List<String> words)
    {
        if (words == null || words.isEmpty())
        {
            return null;
        }
        double[] center = null;
        for (String word : words)
        {
            center = sum(center, wordMap.get(word));
        }
        if (center == null)
        {
            return Collections.emptySet();
        }
        int resultSize = Math.min(wordMap.size(), topNSize);
        TreeSet<WordEntry> result = new TreeSet<>();
        double min = Double.MIN_VALUE;
        for (Map.Entry<String, double[]> entry : wordMap.entrySet())
        {
            String key = entry.getKey();
            if (key == null || words.contains(key))
            {
                continue;
            }
            double[] vector = entry.getValue();
            double dist = 0;
            for (int i = 0; i < vector.length; i++)
            {
                dist += center[i] * vector[i];
            }
            if (dist > min)
            {
                result.add(new WordEntry(entry.getKey(), dist));
                if (resultSize < result.size())
                {
                    result.pollLast();
                    min = result.last().score;
                }
            }
        }
        // result.pollFirst();
        return result;
    }

    private double[] sum(double[] center, double[] fs)
    {
        if (center == null && fs == null)
        {
            return null;
        }
        if (fs == null)
        {
            return center;
        }
        if (center == null)
        {
            return Arrays.copyOf(fs, fs.length);
        }
        for (int i = 0; i < fs.length; i++)
        {
            center[i] += fs[i];
        }
        return center;
    }

    /**
     * Get the word vector for a given word.
     *
     * @param word The word for which to get the vector.
     * @return The word vector as an array of floats.
     */
    public double[] getWordVector(String word)
    {
        return wordMap.get(word);
    }

    public static double readDouble(InputStream is) throws IOException
    {
        byte[] bytes = new byte[8];
        is.read(bytes);
        return getDouble(bytes);
    }

    /**
     * Convert a byte array to a double value.
     *
     * @param b The byte array to convert.
     * @return The double value represented by the byte array.
     */
    public static double getDouble(byte[] b)
    {
        ByteBuffer buffer = ByteBuffer.wrap(b);
        return buffer.getDouble();
    }

    /**
     * 读取一个字符串
     *
     * @param dis DataInputStream对象，用于读取字节流
     * @return 从字节流中读取的字符串
     * @throws IOException 如果在读取过程中发生I/O错误
     */
    private static String readString(DataInputStream dis) throws IOException
    {
        byte[] bytes = new byte[MAX_SIZE];
        byte b = dis.readByte();
        int i = -1;
        StringBuilder sb = new StringBuilder();
        while (b != 32 && b != 10)
        {
            i++;
            bytes[i] = b;
            b = dis.readByte();
            if (i == 49)
            {
                sb.append(new String(bytes));
                i = -1;
                bytes = new byte[MAX_SIZE];
            }
        }
        sb.append(new String(bytes, 0, i + 1));
        return sb.toString();
    }

    public int getTopNSize()
    {
        return topNSize;
    }

    public void setTopNSize(int topNSize)
    {
        this.topNSize = topNSize;
    }

    public HashMap<String, double[]> getWordMap()
    {
        return wordMap;
    }

    public int getWords()
    {
        return words;
    }

    public int getSize()
    {
        return size;
    }

}
