package example;

import classifier.BernoulliClassifier;
import classifier.Classifier;
import classifier.NaiveBayesClassifier;
import lemmatizer.StanfordLemmatizer;

import java.io.*;
import java.nio.file.Files;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class RunnableExample {
    public static final String trainFile = "E:\\Навчання\\4 - КУРС\\Диплом\\JSpamFiltering-master\\dataset\\twitter-databig10\\train.txt";
    public static final String testFile = "E:\\Навчання\\4 - КУРС\\Диплом\\JSpamFiltering-master\\dataset\\twitter-databig10\\test.txt";
    public static final String resultFile = "E:\\Навчання\\4 - КУРС\\Диплом\\JSpamFiltering-master\\dataset\\result.txt";
    public static final String stopFile = "E:\\Навчання\\4 - КУРС\\Диплом\\JSpamFiltering-master\\dataset\\stopwords.txt";

    private static int truePos;
    private static int falsePos;
    private static int trueNeg;
    private static int falseNeg;

    private static Set<String> stopwords = new HashSet<>();

    private static final StanfordLemmatizer lemmatizer = new StanfordLemmatizer();
    static final BernoulliClassifier bernoulli = new BernoulliClassifier();
    static final NaiveBayesClassifier naiveBayes = new NaiveBayesClassifier();

    static Pattern pattern = Pattern.compile("[a-zA-z-]{2,20}");

    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(new File(stopFile)));
        String line;
        while ((line = reader.readLine()) != null) {
            stopwords.add(line);
        }

        train(bernoulli);
        train(naiveBayes);
        test(naiveBayes, null);
        test(bernoulli, null);
    }

    public static void gmoTest(Classifier classifier) throws IOException {
        String trainPos = "E:\\Навчання\\4 - КУРС\\Диплом\\Data\\GMOHedging_v1.0\\processed_pro_GMO.txt";
        String trainNeg = "E:\\Навчання\\4 - КУРС\\Диплом\\Data\\GMOHedging_v1.0\\processed_anti_GMO.txt";
        String test = "E:\\Навчання\\4 - КУРС\\Диплом\\Data\\GMOHedging_v1.0\\processed_test.txt";

        Scanner scanner = new Scanner(new File(trainNeg));
        List<List<String>> positive = new ArrayList<List<String>>();
        List<List<String>> negative = new ArrayList<List<String>>();
        while (scanner.hasNext()) {
            List<String> words = new ArrayList<>();
            Matcher matcher = pattern.matcher(scanner.nextLine());
            while (matcher.find()) {
                String word = matcher.group().toLowerCase();
                if (!stopwords.contains(word)) {
                    words.add(word);
                }
            }
            words = lemmatizer.lemmatize(words);
            negative.add(words);
        }

        scanner = new Scanner(new File(trainPos));
        while (scanner.hasNext()) {
            List<String> words = new ArrayList<>();
            Matcher matcher = pattern.matcher(scanner.nextLine());
            while (matcher.find()) {
                String word = matcher.group().toLowerCase();
                if (!stopwords.contains(word)) {
                    words.add(word);
                }
            }
            words = lemmatizer.lemmatize(words);
            positive.add(words);
        }
//        trainClassifier(positive, 1);
//        trainClassifier(negative, 2);

        trainClassifier(classifier, positive, 1);
        trainClassifier(classifier, negative, 2);
        classifier.applyTraining();
        test(classifier, test);
    }

    public static void train(Classifier classifier) throws IOException {
        Scanner scanner;
        /*
         * The classifier can learn from classifications that are handed over
         * to the learn methods. Imagin a tokenized text as follows. The tokens
         * are the text's features. The category of the text will either be
         * positive or negative.
         */
        //scanner = new Scanner(new File(trainFile));
        List<String> lines = Files.readAllLines(new File(trainFile).toPath());
        List<List<String>> positive = new ArrayList<List<String>>();
        List<List<String>> negative = new ArrayList<List<String>>();

        int i = 0;
        while (i < lines.size()) {
            String line = lines.get(i++);
            int emotion = line.charAt(0) - 48;
            if (emotion == 1) {
                List<String> words = new ArrayList<>();
                Matcher matcher = pattern.matcher(line);
                while (matcher.find()) {
                    String word = matcher.group().toLowerCase();
                    if (!stopwords.contains(word)) {
                        words.add(word);
                    }
                }
                //if (words.size() > 1) {
                    words = lemmatizer.lemmatize(words);
                    positive.add(words);
                //}
            } else {
                List<String> words = new ArrayList<>();
                Matcher matcher = pattern.matcher(line);
                while (matcher.find()) {
                    String word = matcher.group().toLowerCase();
                    if (!stopwords.contains(word)) {
                        words.add(word);
                    }
                }
                //if (words.size() > 1) {
                    words = lemmatizer.lemmatize(words);
                    negative.add(words);
                //}
            }
        }

        trainClassifier(classifier, positive, 1);
        trainClassifier(classifier, negative, 2);
        classifier.applyTraining();
        System.out.println(classifier.toString() + " training complete");
    }


    public static void trainClassifier(Classifier classifier, List<List<String>> text, int category) {
        for (List<String> line : text) {
            classifier.learn(category, line);
        }
    }

    public static void test(Classifier classifier, String filepath) throws IOException {
        List<String> lines;
        if (filepath != null) {
            lines = Files.readAllLines(new File(filepath).toPath());
        } else {
            lines = Files.readAllLines(new File(testFile).toPath());
        }
        List<String> words = new ArrayList<>();
        int correct = 0;
        int wrong = 0;
        int i = 0;

        while (i < lines.size()) {
            String line = lines.get(i++);
            int emotion = line.charAt(0) - 48;

            Matcher matcher = pattern.matcher(line);
            while (matcher.find()) {
                String word = matcher.group().toLowerCase();
                if (!stopwords.contains(word)) {
                    words.add(word);
                }
            }
            words = lemmatizer.lemmatize(words);

            int result = classifier.classify(words);
            checkResult(emotion, result);
            //System.out.println(emotion + " " + result);
            if (emotion == result) {
                correct++;
            } else {
                wrong++;
            }
            words.clear();
        }
        System.out.println(correct + " " + wrong);
        double result = (correct * 1.0) / (correct + wrong);
        System.out.println(result);

        BufferedWriter writer = new BufferedWriter(new FileWriter(new File(resultFile), true));
        writer.write(classifier.toString() + "  lemmStop  " + "twitterBig   " + "accuracy: " + getAccuracy()
                + " precision: " + getPrecision() + " recall: " + getRecall());
        writer.newLine();
        writer.close();
    }

    public static void checkResult(int emotion, int result) {
        if (emotion == 1 && result == 1) {
            truePos++;
        } else if (emotion == 1 && result == 2) {
            falsePos++;
        } else if (emotion == 2 && result == 1) {
            falseNeg++;
        } else {
            trueNeg++;
        }
    }

    public static double getAccuracy()
    {
        return ((double)(truePos + trueNeg)) / ((double)trueNeg + truePos + falseNeg + falsePos);
    }

    public static double getPrecision()
    {
        //if((TruePositives + FalsePositives)==0) return 0;
        return  ((double)truePos / (double)(truePos + falsePos));
    }

    public static double getRecall()
    {
        //if((TruePositives + FalseNegatives)==0) return 0;
        return ((double)truePos / (double)(truePos + falseNeg));
    }

    public static double getFMeasure()
    {
        double precision = getPrecision();
        double recall = getRecall();

        if(precision + recall == 0) return 0;
        return (2 * precision * recall) / (precision + recall);
    }
}
