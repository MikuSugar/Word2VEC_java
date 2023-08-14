package me.mikusugar.word2vec;

import com.google.common.base.Preconditions;
import it.unimi.dsi.fastutil.ints.IntList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.file.Files;

public class SkipGram extends Word2Vec
{
    private static final Logger logger = LoggerFactory.getLogger(SkipGram.class);

    public SkipGram()
    {

    }

    private void trainModel(File file) throws IOException
    {

        final long startTime = System.currentTimeMillis();
        this.alpha = startingAlpha;
        try (BufferedReader br = new BufferedReader(new InputStreamReader(Files.newInputStream(file.toPath()))))
        {
            String line;
            wordCount = 0;
            lastWordCount = 0;
            wordCountActual = 0;
            while ((line = br.readLine()) != null)
            {
                updateLearRate();
                String[] strs = line.split("[\\s　]+");
                wordCount += strs.length;
                final IntList sentence = getSentence(strs);

                for (int index = 0; index < sentence.size(); index++)
                {
                    skipGram(index, sentence, random.nextInt(window));
                }
            }
            logger.info("Vocab size: " + word2idx.size());
            logger.info("Words in train file: " + trainWordsCount);
            logger.info("success train over! take time:{}ms.", System.currentTimeMillis() - startTime);
        }
    }

    /**
     * skipGram
     *
     * @param index    the index of the word in the sentence
     * @param sentence the sentence containing the word
     * @param b        the context window size
     */
    private void skipGram(int index, IntList sentence, int b)
    {
        int input = sentence.getInt(index);
        int a, c;
        for (a = b; a < window * 2 + 1 - b; a++)
        {
            if (a == window)
            {
                continue;
            }
            c = index - window + a;
            if (c < 0 || c >= sentence.size())
            {
                continue;
            }

            double[] neu1e = new double[layerSize];// 误差项
            int we = sentence.getInt(c);

            negativeSampling(input, we, neu1e);

            // Learn weights input -> hidden
            for (int j = 0; j < layerSize; j++)
            {
                syn0[input][j] += neu1e[j];
            }
        }

    }

    private void negativeSampling(int input, int we, double[] neu1e)
    {
        int target;
        int label;
        for (int d = 0; d < negative + 1; d++)
        {
            if (d == 0)
            {
                target = we;
                label = 1;
            }
            else
            {
                target = negativeSampling.next();
                if (target == input)
                {
                    continue;
                }
                label = 0;
            }
            final double g = getG(input, target, label);
            for (int i = 0; i < layerSize; i++)
            {
                neu1e[i] += g * syn1[target][i];
                syn1[target][i] += g * syn0[input][i];
            }
        }
    }

    @Override
    public void fitFile(String filePath) throws IOException
    {
        File file = new File(filePath);
        Preconditions.checkArgument(file.isFile());

        createExpTable();
        readVocab(file);
        initNet();
        initNegative();
        trainModel(file);
    }

}
