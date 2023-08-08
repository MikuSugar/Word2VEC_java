package me.mikusugar.node2vec;

import com.ansj.vec.Learn;
import com.ansj.vec.domain.Neuron;
import com.ansj.vec.domain.WordNeuron;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author mikusugar
 * @version 1.0, 2023/8/3 17:06
 * @description
 */
public class Node2VecLearn
{
    /*
    超参数p，控制遍历的返回前一个节点的概率。
    */
    private double p = 1;

    /*
    超参数q，控制节点继续向前行的概率。
     */
    private double q = 1;

    /*
    每条路线上节点的数量
    */
    private int walkLength = 20;

    /*
    从头到尾反复遍历次数
    */
    private int numWalks = 10;

    /**
     * 训练多少个特征
     */
    private int layerSize = 242;

    /**
     * 上下文窗口大小
     */
    private int window = 5;

    private double sample = 1e-3;

    private double alpha = 0.025;

    private final Learn learn;

    private Map<Integer, double[]> nodeMap;

    private boolean isCbow = false;

    private int MAX_EAP = 6;

    public Node2VecLearn(double p, double q, int walkLength, int numWalks, int layerSize, int window, double sample,
            double alpha, boolean isCbow, int MAX_EAP)
    {
        this.p = p;
        this.q = q;
        this.walkLength = walkLength;
        this.numWalks = numWalks;
        this.layerSize = layerSize;
        this.window = window;
        this.sample = sample;
        this.alpha = alpha;
        this.isCbow = isCbow;
        this.MAX_EAP = MAX_EAP;
        this.learn = new Learn(isCbow, layerSize, window, alpha, sample);
        this.learn.setMAX_EXP(MAX_EAP);
        check();
    }

    private void check()
    {
        if (this.window > this.walkLength)
        {
            throw new IllegalArgumentException("window参数必须小于walkLength");
        }
    }

    public Node2VecLearn()
    {
        this.learn = new Learn(false, layerSize, window, alpha, sample);
    }

    public void lean(Graph graph)
    {
        check();
        final RandomWalk randomWalk = new RandomWalk(p, q, walkLength, numWalks, graph);
        final List<int[]> simulateWalks = randomWalk.simulateWalks();
        learn.learnData(simulateWalks);
        final Map<String, Neuron> wordMap = learn.getWordMap();
        nodeMap = new HashMap<>(wordMap.size());

        wordMap.forEach((k, v) -> nodeMap.put(Integer.parseInt(k), ((WordNeuron)v).syn0));
    }

    public void saveMode(String path) throws IOException
    {
        int dimension = -1;
        for (double[] v : nodeMap.values())
        {
            dimension = v.length;
            break;
        }
        if (dimension == -1)
        {
            throw new IllegalArgumentException("nodeMap is empty!");
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(path)))
        {
            writer.write(nodeMap.size() + " " + dimension + System.lineSeparator());
            for (Map.Entry<Integer, double[]> entry : nodeMap.entrySet())
            {
                int node = entry.getKey();
                double[] vector = entry.getValue();
                writer.write(node + "");
                for (double v : vector)
                {
                    writer.write(" " + v);
                }
                writer.write(System.lineSeparator());
            }
        }
    }

    public void setP(double p)
    {
        this.p = p;
    }

    public void setQ(double q)
    {
        this.q = q;
    }

    public void setWalkLength(int walkLength)
    {
        this.walkLength = walkLength;
    }

    public void setNumWalks(int numWalks)
    {
        this.numWalks = numWalks;
    }

    public void setLayerSize(int layerSize)
    {
        this.layerSize = layerSize;
        this.learn.setLayerSize(layerSize);
    }

    public void setWindow(int window)
    {
        this.window = window;
        this.learn.setWindow(window);
    }

    public void setSample(double sample)
    {
        this.sample = sample;
        this.learn.setSample(sample);
    }

    public void setAlpha(double alpha)
    {
        this.alpha = alpha;
        this.learn.setAlpha(alpha);
    }

    public void setCbow(boolean cbow)
    {
        isCbow = cbow;
        this.learn.setIsCbow(isCbow);
    }

    public void setMAX_EAP(int MAX_EAP)
    {
        this.MAX_EAP = MAX_EAP;
        this.learn.setMAX_EXP(MAX_EAP);
    }
}
