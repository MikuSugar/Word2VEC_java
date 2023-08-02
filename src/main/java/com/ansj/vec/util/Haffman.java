package com.ansj.vec.util;

import java.util.Collection;
import java.util.Objects;
import java.util.TreeSet;

import com.ansj.vec.domain.HiddenNeuron;
import com.ansj.vec.domain.Neuron;

/**
 * Constructs Huffman coding tree.
 * <p>
 * This class is responsible for constructing a Huffman coding tree.
 * The tree is built by merging neurons from the given collection until only one neuron remains.
 * Neurons are merged based on their frequency values.
 * <p>
 * The constructor requires the size of the hidden layer as a parameter.
 * The hidden layer size determines the number of neurons in the hidden layer of the tree.
 *
 * @since 1.0
 */
public class Haffman
{
    private final int layerSize;

    public Haffman(int layerSize)
    {
        this.layerSize = layerSize;
    }

    private final TreeSet<Neuron> set = new TreeSet<>();

    public void make(Collection<Neuron> neurons)
    {
        set.addAll(neurons);
        while (set.size() > 1)
        {
            merger();
        }
    }

    private void merger()
    {
        HiddenNeuron hn = new HiddenNeuron(layerSize);
        Neuron min1 = set.pollFirst();
        Neuron min2 = set.pollFirst();
        hn.category = Objects.requireNonNull(min2).category;
        hn.freq = Objects.requireNonNull(min1).freq + min2.freq;
        min1.parent = hn;
        min2.parent = hn;
        min1.code = 0;
        min2.code = 1;
        set.add(hn);
    }

}
