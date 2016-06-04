package classifier;

import java.util.*;

/**
 * Created by Vlad on 30.05.2016.
 */
public class BernoulliClassifier extends Classifier {
    private static final int categories = 2;
    private int docsCount;

    private int[] categoryCount = new int[categories];
    private double[] priors = new double[categories];

    private Set<String> vocabulary = new HashSet<>();

    private Map<Integer, HashMap<String, Integer>> featureCountPerCategory;
    private Map<Integer, HashMap<String, Double>> conditionalProbabilities;


    public BernoulliClassifier() {
        featureCountPerCategory = new HashMap<>();
        featureCountPerCategory.put(1, new HashMap<>());
        featureCountPerCategory.put(2, new HashMap<>());

        conditionalProbabilities = new HashMap<>();

    }

    public void learn(int category, List<String> words) {
        docsCount++;
        Set<String> uniqueWords = new HashSet<>(words);

        HashMap<String, Integer> categoryFeatures = featureCountPerCategory.get(category);
        for (String word : uniqueWords) {
            vocabulary.add(word);

            if (categoryFeatures.containsKey(word)) {
                int count = categoryFeatures.get(word);
                categoryFeatures.replace(word, count + 1);
            } else {
                categoryFeatures.put(word, 1);
            }
        }
        categoryCount[category - 1]++;
        featureCountPerCategory.replace(category, categoryFeatures);
    }

    public void applyTraining() {
        for (int i = 1; i <= categories; i++) {
            HashMap<String, Integer> categoryFeatures = featureCountPerCategory.get(i);
            HashMap<String, Double> conditionals = new HashMap<>();
            for (String word : vocabulary) {

                if (categoryFeatures.containsKey(word)) {
                    conditionals.put(word, (categoryFeatures.get(word) + 1.0) / (categoryCount[i - 1] + categories));
                } else {
                    conditionals.put(word, (1.0) / (categoryCount[i - 1] + categories));
                }


            }
            conditionalProbabilities.put(i, conditionals);
            priors[i - 1] = categoryCount[i - 1] * 1.0 / docsCount;
        }

    }

    public int classify(List<String> words) {
        double[] results = new double[categories];

        for (int i = 1; i <= categories; i++) {
            double result = Math.log(priors[i - 1]);
            HashMap<String, Double> conditionals = conditionalProbabilities.get(i);

            for (String word : vocabulary) {
                if (words.contains(word)) {
                    result += Math.log(conditionals.get(word));
                } else {
                    //System.out.println(conditionals.get(word));
                    result += Math.log(1 - conditionals.get(word));
                }
            }
            results[i - 1] = result;
        }
        return results[0] >= results[1] ? 1 : 2;
    }

    @Override
    public String toString() {
        return "BernoulliClassifier: ";
    }
}
