package com.ansj.vec.domain;

/**
 * Neuron is an abstract class that represents a neuron in a neural network, implementing the Comparable interface to
 * allow comparison between neurons.
 */
public abstract class Neuron implements Comparable<Neuron>
{
    public double freq;

    public Neuron parent;

    public int code;

    // 语料预分类
    public int category = -1;

    @Override
    public int compareTo(Neuron neuron)
    {
        if (this.category == neuron.category)
        {
            if (this.freq > neuron.freq)
            {
                return 1;
            }
            else
            {
                return -1;
            }
        }
        else if (this.category > neuron.category)
        {
            return 1;
        }
        else
        {
            return -1;
        }
    }
}
