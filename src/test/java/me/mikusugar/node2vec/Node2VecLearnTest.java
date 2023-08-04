package me.mikusugar.node2vec;

import org.junit.Test;

import java.io.IOException;

/**
 * @author mikusugar
 * @version 1.0, 2023/8/4 09:42
 * @description
 */
public class Node2VecLearnTest
{
    @Test
    public void learnKarateTest() throws IOException
    {
        final String path = HelpTestUtils.getResourcePath() + "/karate.edgelist";
        final Graph graph = ParseUtils.edgeListFile2Graph(path);
        Node2VecLearn learn = new Node2VecLearn();
        learn.lean(graph);
        learn.saveMode("graph_model.bin");

    }
}
