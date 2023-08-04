package me.mikusugar.node2vec;

import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;

/**
 * @author mikusugar
 * @version 1.0, 2023/8/4 09:42
 * @description
 */
public class Node2VecTest
{
    private final String karateModelPath = "karate_model.bin";

    @Test
    public void learnKarateTest() throws IOException
    {
        final String path = HelpTestUtils.getResourcePath() + "/karate.edgelist";
        final Graph graph = ParseUtils.edgeListFile2Graph(path);
        Node2VecLearn learn = new Node2VecLearn();
        learn.lean(graph);

        learn.saveMode(karateModelPath);
    }

    @Test
    public void node2vecKarateTest() throws IOException
    {
        Node2Vec node2Vec = new Node2Vec();
        node2Vec.loadJavaModel(karateModelPath);
        System.out.println("22:" + node2Vec.closestNodes(22));

        System.out.println("27 30 3" + node2Vec.closestNodes(Arrays.asList(27, 30, 3)));
    }

}
