package com.ansj.vec.util;

import java.io.IOException;
import java.util.*;
import java.util.Map.Entry;

import com.ansj.vec.Word2vec;

/**
 * keanmeans聚类
 *
 * @author ansj
 */
public class WordKmeans
{

    public static void main(String[] args) throws IOException
    {
        Word2vec vec = new Word2vec();
        vec.loadGoogleModel("vectors.bin");
        System.out.println("load model ok!");
        WordKmeans wordKmeans = new WordKmeans(vec.getWordMap(), 50, 50);
        Classes[] explain = wordKmeans.explain();

        for (int i = 0; i < explain.length; i++)
        {
            System.out.println("--------" + i + "---------");
            System.out.println(explain[i].getTop(10));
        }

    }

    private final HashMap<String, double[]> wordMap;

    private final int iter;

    private final Classes[] cArray;

    public WordKmeans(HashMap<String, double[]> wordMap, int clcn, int iter)
    {
        this.wordMap = wordMap;
        this.iter = iter;
        cArray = new Classes[clcn];
    }

    public Classes[] explain()
    {
        //first 取前clcn个点
        Iterator<Entry<String, double[]>> iterator = wordMap.entrySet().iterator();
        for (int i = 0; i < cArray.length; i++)
        {
            Entry<String, double[]> next = iterator.next();
            cArray[i] = new Classes(i, next.getValue());
        }

        for (int i = 0; i < iter; i++)
        {
            for (Classes classes : cArray)
            {
                classes.clean();
            }

            iterator = wordMap.entrySet().iterator();
            while (iterator.hasNext())
            {
                Entry<String, double[]> next = iterator.next();
                double miniScore = Double.MAX_VALUE;
                double tempScore;
                int classesId = 0;
                for (Classes classes : cArray)
                {
                    tempScore = classes.distance(next.getValue());
                    if (miniScore > tempScore)
                    {
                        miniScore = tempScore;
                        classesId = classes.id;
                    }
                }
                cArray[classesId].putValue(next.getKey(), miniScore);
            }

            for (Classes classes : cArray)
            {
                classes.updateCenter(wordMap);
            }
            System.out.println("iter " + i + " ok!");
        }

        return cArray;
    }

    public static class Classes
    {
        private final int id;

        private final double[] center;

        public Classes(int id, double[] center)
        {
            this.id = id;
            this.center = center.clone();
        }

        Map<String, Double> values = new HashMap<>();

        public double distance(double[] value)
        {
            double sum = 0;
            for (int i = 0; i < value.length; i++)
            {
                sum += (center[i] - value[i]) * (center[i] - value[i]);
            }
            return sum;
        }

        public void putValue(String word, double score)
        {
            values.put(word, score);
        }

        /**
         * 重新计算中心点
         *
         * @param wordMap wordMap
         */
        public void updateCenter(HashMap<String, double[]> wordMap)
        {
            Arrays.fill(center, 0);
            double[] value;
            for (String keyWord : values.keySet())
            {
                value = wordMap.get(keyWord);
                for (int i = 0; i < value.length; i++)
                {
                    center[i] += value[i];
                }
            }
            for (int i = 0; i < center.length; i++)
            {
                center[i] = center[i] / values.size();
            }
        }

        /**
         * 清空历史结果
         */
        public void clean()
        {
            values.clear();
        }

        /**
         * 取得每个类别的前n个结果
         *
         */
        public List<Entry<String, Double>> getTop(int n)
        {
            List<Map.Entry<String, Double>> arrayList = new ArrayList<>(values.entrySet());
            arrayList.sort(Comparator.comparingDouble(Entry::getValue));
            int min = Math.min(n, arrayList.size() - 1);
            if (min <= 1)
                return Collections.emptyList();
            return arrayList.subList(0, min);
        }

    }

}
