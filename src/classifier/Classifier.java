package classifier;

import java.util.List;

/**
 * Created by Vlad on 01.06.2016.
 */
public abstract class Classifier {

    public abstract void learn(int category, List<String> words);
    public abstract int classify(List<String> words);
    public abstract void applyTraining();
}
