package me.mikusugar.node2vec;

import com.ansj.vec.Word2vec;
import com.ansj.vec.domain.WordEntry;

import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * @description
 * @author mikusugar
 * @version 1.0, 2023/8/4 10:28
 */
public class Node2Vec
{
    private final Word2vec word2vec;

    public Node2Vec()
    {
        this.word2vec = new Word2vec();
    }

    public void loadJavaModel(String path) throws IOException
    {
        this.word2vec.loadJavaModel(path);
    }

    public Set<WordEntry> closestNodes(Collection<Integer> nodes)
    {
        final List<String> words = nodes.stream().map(String::valueOf).collect(Collectors.toList());
        return word2vec.distance(words);
    }

    public Set<WordEntry> closestNodes(int node)
    {
        return word2vec.distance(String.valueOf(node));
    }

}
