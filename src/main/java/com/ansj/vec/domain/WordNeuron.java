package com.ansj.vec.domain;

import java.util.*;

public class WordNeuron extends Neuron
{
    public String name;

    public double[] syn0; // input->hidden

    public List<Neuron> neurons = null;// 路径神经元

    public int[] codeArr = null;

    public double[] negativeSyn1;



    public void makeNeurons()
    {
        if (neurons != null)
        {
            return;
        }
        Neuron neuron = this;
        neurons = new ArrayList<>();
        while ((neuron = neuron.parent) != null)
        {
            neurons.add(neuron);
        }
        if (neurons.isEmpty())
        {
            return;
        }
        Collections.reverse(neurons);
        codeArr = new int[neurons.size()];

        for (int i = 1; i < neurons.size(); i++)
        {
            codeArr[i - 1] = neurons.get(i).code;
        }
        codeArr[codeArr.length - 1] = this.code;
    }

    public WordNeuron(String name, double freq, int layerSize)
    {
        this.name = name;
        this.freq = freq;
        this.syn0 = new double[layerSize];
        Random random = new Random();
        for (int i = 0; i < syn0.length; i++)
        {
            syn0[i] = (random.nextDouble() - 0.5) / layerSize;
        }
    }

    /**
     * Creates a WordNeuron object for supervised creation of Hoffman tree.
     *
     * @param name      The name of the WordNeuron.
     * @param freq      The frequency of the WordNeuron.
     * @param category  The category of the WordNeuron.
     * @param layerSize The size of the layer.
     */
    public WordNeuron(String name, double freq, int category, int layerSize)
    {
        this.name = name;
        this.freq = freq;
        this.syn0 = new double[layerSize];
        this.category = category;
        Random random = new Random();
        for (int i = 0; i < syn0.length; i++)
        {
            syn0[i] = (random.nextDouble() - 0.5) / layerSize;
        }
    }

    @Override
    public int compareTo(Neuron neuron)
    {
        if (this.category == neuron.category)
        {
            if (neuron instanceof WordNeuron && this.freq == neuron.freq)
            {
                return this.name.compareTo(((WordNeuron)neuron).name);
            }
            return Double.compare(this.freq, neuron.freq);
        }
        return Integer.compare(this.category, neuron.category);
    }

}
