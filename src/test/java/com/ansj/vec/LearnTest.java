package com.ansj.vec;

import java.io.File;
import java.io.IOException;

import com.ansj.vec.Learn;

public class LearnTest
{

    public static void main(String[] args) throws IOException
    {
        Learn learn = new Learn();
        learn.setLayerSize(200);
        learn.setMAX_EXP(10);
        learn.learnFile(new File("corpus.txt"));
        learn.saveModel(new File("model.bin"));
    }
}
