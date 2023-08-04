package com.ansj.vec.domain;

public class WordEntry implements Comparable<WordEntry>
{
    public String name;

    public double score;

    public WordEntry(String name, double score)
    {
        this.name = name;
        this.score = score;
    }

    @Override
    public String toString()
    {
        return this.name + "\t" + score;
    }

    @Override
    public int compareTo(WordEntry o)
    {
        if (this.score < o.score)
        {
            return 1;
        }
        else
        {
            return -1;
        }
    }

}
