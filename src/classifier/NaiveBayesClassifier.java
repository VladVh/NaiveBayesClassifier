package classifier;

import java.util.*;

/**
 * Created by Vlad on 01.06.2016.
 */
public class NaiveBayesClassifier extends Classifier{
    private static final int categories = 2;
    private int docsCount;


    private int[] categoryCount = new int[categories];
    private int[] wordsPerCategoryCount = new int[categories];
    private double[] priors = new double[categories];

    private Set<String> vocabulary = new HashSet<>();

    private Map<Integer, HashMap<String, Integer>> tokenCountPerCategory;
    private Map<Integer, HashMap<String, Double>> conditionalProbabilities;


    public NaiveBayesClassifier() {
        tokenCountPerCategory = new HashMap<>();
        tokenCountPerCategory.put(1, new HashMap<>());
        tokenCountPerCategory.put(2, new HashMap<>());

        conditionalProbabilities = new HashMap<>();
    }

    public void learn(int category, List<String> words) {
        docsCount++;
        wordsPerCategoryCount[category - 1] += words.size();

        HashMap<String, Integer> categoryFeatures = tokenCountPerCategory.get(category);
        for (String word : words) {
            vocabulary.add(word);

            if (categoryFeatures.containsKey(word)) {
                int count = categoryFeatures.get(word);
                categoryFeatures.replace(word, count + 1);
            } else {
                categoryFeatures.put(word, 1);
            }
        }
        categoryCount[category - 1]++;
        tokenCountPerCategory.replace(category, categoryFeatures);
    }

    public void applyTraining() {
        for (int i = 0; i < categories; i++) {
            HashMap<String, Integer> categoryTokens = tokenCountPerCategory.get(i + 1);
            HashMap<String, Double> conditionals = new HashMap<>();
            for (String word : vocabulary) {

                if (categoryTokens.containsKey(word)) {
                    conditionals.put(word, (categoryTokens.get(word) + 1.0) / (wordsPerCategoryCount[i] + vocabulary.size()));
                } else {
                    conditionals.put(word, (1.0) / (wordsPerCategoryCount[i] + vocabulary.size()));
                }

            }
            conditionalProbabilities.put(i + 1, conditionals);
            priors[i] = categoryCount[i] * 1.0 / docsCount;
        }

    }

    public int classify(List<String> words) {
        double[] results = new double[categories];

        for (int i = 1; i <= categories; i++) {
            double result = Math.log(priors[i - 1]);
            HashMap<String, Double> conditionals = conditionalProbabilities.get(i);

            for (String word : words) {
                //if (vocabulary.contains(word)) {
                Double cond = conditionals.get(word);
                if (cond != null) {
                result += Math.log(cond);
                } else {
                    result += Math.log(1.0 / (wordsPerCategoryCount[i - 1] + vocabulary.size()));
                }
            }
            results[i - 1] = result;
        }
        return results[0] >= results[1] ? 1 : 2;
    }

    @Override
    public String toString() {
        return "NaiveBayesClassifier: ";
    }
}
