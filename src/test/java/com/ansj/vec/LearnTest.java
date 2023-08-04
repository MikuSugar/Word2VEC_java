package com.ansj.vec;

import java.io.File;
import java.io.IOException;

import com.ansj.vec.Learn;
import me.mikusugar.node2vec.HelpTestUtils;

public class LearnTest
{

    public static void main(String[] args) throws IOException
    {
        final String corpusFilePath = HelpTestUtils.getResourcePath() + "/corpus.txt";
        Learn learn = new Learn();
        learn.setLayerSize(200);
        learn.setMAX_EXP(10);

        learn.learnFile(new File(corpusFilePath));
        learn.saveModel(new File("model.bin"));
    }
}
