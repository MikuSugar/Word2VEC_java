package com.ansj.vec;

import me.mikusugar.node2vec.HelpTestUtils;

import java.io.File;
import java.io.IOException;

public class LearnTest
{

    public static void main(String[] args) throws IOException
    {
        final String corpusFilePath = HelpTestUtils.getResourcePath() + "/corpus.txt";
        Learn learn = new Learn();
        learn.setLayerSize(200);
        learn.setMAX_EXP(6);
//        learn.setNegative(20);

        learn.learnFile(new File(corpusFilePath));
        learn.saveModel(new File("model.bin"));
    }
}
