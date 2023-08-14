package me.mikusugar.node2vec;

import me.mikusugar.HelpTestUtils;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * @author mikusugar
 * @version 1.0, 2023/8/4 09:42
 * @description
 */
public class Node2VecTest
{
    private final String karateModelPath = "karate_model.emb";

    @Test
    public void learnKarateTest() throws IOException
    {
        final String path = HelpTestUtils.getResourcePath() + "/karate.edgelist";
        final Graph graph = ParseUtils.edgeListFile2Graph(path, false);
        Node2VecLearn learn = new Node2VecLearn(1, 1, 80, 10, 128, 10, 1e-3, 0.025, false, 6);
        learn.lean(graph);
        learn.setNegative(10);
        learn.saveMode(karateModelPath);
    }

    @Test
    public void node2vecKarateTest() throws IOException
    {
        Node2Vec node2Vec = new Node2Vec();
        node2Vec.loadEmbModel(karateModelPath);

        int node = 22;
        System.out.println(node + ":" + node2Vec.closestNodes(node));

        final List<Integer> list = Arrays.asList(1, 22, 24);
        System.out.println(list + "::" + node2Vec.closestNodes(list));
        node2Vec.getNodeMap().forEach((k, v) -> System.out.println(k + "::" + Arrays.toString(v)));
    }

}
