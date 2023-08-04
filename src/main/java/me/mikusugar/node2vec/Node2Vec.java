package me.mikusugar.node2vec;

import com.ansj.vec.Word2vec;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * @description
 * @author mikusugar
 * @version 1.0, 2023/8/4 10:28
 */
public class Node2Vec
{
    private final Word2vec word2vec;

    private final int topNSize = 10;

    public Node2Vec()
    {
        this.word2vec = new Word2vec();
    }

    private Map<Integer, double[]> nodeMap;

    public void loadJavaModel(String path) throws IOException
    {
        this.word2vec.loadJavaModel(path);
        this.nodeMap = word2vec.getWordMap().entrySet().stream()
                .collect(Collectors.toMap(e -> Integer.parseInt(e.getKey()), Map.Entry::getValue));
    }

    public List<NodeEntry> closestNodes(Collection<Integer> nodes)
    {
        if (nodes == null || nodes.isEmpty())
        {
            throw new IllegalArgumentException("nodes not empty!");
        }
        final double[] vectorSum = sumVector(nodes);
        final int size = Math.min(nodeMap.size() - 1, topNSize);
        Set<Integer> nodesSet = new HashSet<>(nodes);
        PriorityQueue<NodeEntry> pq = new PriorityQueue<>();
        nodeMap.forEach((curNode, curVector) ->
        {
            if (!nodesSet.contains(curNode))
            {
                final double score = calculateCosineSimilarity(vectorSum, curVector);
                pq.add(new NodeEntry(curNode, score));
                if (pq.size() > size)
                {
                    pq.poll();
                }
            }
        });
        return pq2list(pq);
    }

    private double[] sumVector(Collection<Integer> nodes)
    {
        double[] res = null;
        for (int node : nodes)
        {
            final double[] vector = nodeMap.get(node);
            if (vector == null)
            {
                throw new IllegalArgumentException("not found " + node);
            }
            if (res == null)
            {
                res = Arrays.copyOf(vector, vector.length);
            }
            else
            {
                for (int i = 0; i < vector.length; i++)
                {
                    res[i] += vector[i];
                }
            }
        }
        return res;
    }

    public List<NodeEntry> closestNodes(int node)
    {
        final double[] vector = nodeMap.get(node);
        if (vector == null)
        {
            throw new IllegalArgumentException("not found:" + node);
        }
        final PriorityQueue<NodeEntry> pq = new PriorityQueue<>();
        final int size = Math.min(nodeMap.size() - 1, topNSize);
        nodeMap.forEach((curNode, curVector) ->
        {
            if (curNode != node)
            {
                final double score = calculateCosineSimilarity(vector, curVector);
                pq.add(new NodeEntry(curNode, score));
                if (pq.size() > size)
                {
                    pq.poll();
                }
            }
        });
        return pq2list(pq);
    }

    private static List<NodeEntry> pq2list(PriorityQueue<NodeEntry> pq)
    {
        final List<NodeEntry> result = new ArrayList<>(pq.size());
        while (!pq.isEmpty())
        {
            result.add(pq.poll());
        }
        Collections.reverse(result);
        return result;
    }

    /**
     * Calculates the cosine similarity between two vectors.
     *
     * @param vectorA the first vector
     * @param vectorB the second vector
     * @return the cosine similarity between the two vectors
     * @throws IllegalArgumentException if the vectors have different dimensions or one or both vectors have zero norm
     */
    private static double calculateCosineSimilarity(double[] vectorA, double[] vectorB)
    {
        if (vectorA.length != vectorB.length)
        {
            throw new IllegalArgumentException("Vectors must have the same dimensions");
        }

        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < vectorA.length; i++)
        {
            dotProduct += vectorA[i] * vectorB[i];
            normA += vectorA[i] * vectorA[i];
            normB += vectorB[i] * vectorB[i];
        }

        if (normA == 0.0 || normB == 0.0)
        {
            throw new IllegalArgumentException("One or both vectors have zero norm");
        }

        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    public Map<Integer, double[]> getNodeMap()
    {
        return nodeMap;
    }

}
