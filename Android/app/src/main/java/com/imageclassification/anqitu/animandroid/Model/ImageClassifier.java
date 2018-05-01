package com.imageclassification.anqitu.animandroid.Model;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Trace;
import android.util.Log;

import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Vector;

public class ImageClassifier implements Classifier {
    private static final String TAG = "ImageClassifier";

    // Only return this many results with at least this confidence.
    private static final int MAX_RESULTS = 3;
//    private static final int BATCH_SIZE = 1;
//    private static final int PIXEL_SIZE = 3;
    private static final float THRESHOLD = 0.1f;

    // Config values.
    private String inputName;
    private String outputName;
    private int inputSize;
//    private int imageMean;
//    private float imageStd;

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;
    private float[] pixels;
    private float[] probs;
    private String[] outputNames;

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface;

    private ImageClassifier() {}


    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     * @param inputSize The input size. A square image of inputSize x inputSize is assumed.
//     * @param imageMean The assumed mean of the image values.
//     * @param imageStd The assumed std of the image values.
     * @param inputName The label of the image input node.
     * @param outputName The label of the output node.
     * @throws IOException
     */
    public static Classifier create(
            AssetManager assetManager,
            String modelFilename,
            String labelFilename,
            int inputSize,
            String inputName,
            String outputName) {
        ImageClassifier c = new ImageClassifier();
        c.inputName = inputName;
        c.outputName = outputName;

        // Read the label names into memory.
        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        Log.i(TAG, "Reading labels from: " + actualFilename);
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(assetManager.open(actualFilename)));
            String line;
            while ((line = br.readLine()) != null) {
                c.labels.add(line);
            }
            br.close();
        } catch (IOException e) {
            throw new RuntimeException("Problem reading label file!" , e);
        }

        c.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

        // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
        final Operation operation = c.inferenceInterface.graphOperation(outputName);
        final int numClasses = (int) operation.output(0).shape().size(1);
        Log.i(TAG, "Read " + c.labels.size() + " labels, output layer size is " + numClasses);

        c.inputSize = inputSize;

        // Pre-allocate buffers.
        c.outputNames = new String[] {outputName};
        c.intValues = new int[inputSize * inputSize];
        c.pixels = new float[inputSize * inputSize * 3];
        c.probs = new float[numClasses];

        return c;
    }


    @Override
    public ArrayList<LabelProb> recognizeImage(Bitmap bitmap) {

        ArrayList<LabelProb> labelProbs = new ArrayList<>();

        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        try {
            bitmap = getResizedBitmap(bitmap, 150, 150);
            bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
            for (int i = 0; i < intValues.length; ++i) {
                final int val = intValues[i];
                pixels[i * 3 + 0] = (((val >> 16) & 0xFF) / 255.0f);
                pixels[i * 3 + 1] = (((val >> 8) & 0xFF) / 255.0f);
                pixels[i * 3 + 2] = ((val & 0xFF) / 255.0f);
            }
            Trace.endSection();

            // Copy the input data into TensorFlow.
            Trace.beginSection("feed");
            inferenceInterface.feed(inputName, pixels, 1, inputSize, inputSize, 3);
            Trace.endSection();

            // Run the inference call.
            Trace.beginSection("run");
            inferenceInterface.run(outputNames, logStats);
            Trace.endSection();

            // Copy the output Tensor back into the output array.
            Trace.beginSection("fetch");
            inferenceInterface.fetch(outputName, probs);
            Trace.endSection();



            for (int i = 0; i < probs.length; ++i) {
                System.out.println(probs[i]);
                System.out.println(labels.get(i));
                labelProbs.add(new LabelProb(probs[i], labels.get(i)));
            }

            Collections.sort(labelProbs, new Comparator<LabelProb>() {
                @Override
                public int compare(LabelProb o1, LabelProb o2) {
                    return Float.compare(o2.getProb(), o1.getProb());
                }
            });


            for (int i = 0; i < probs.length; ++i) {
                System.out.println(labelProbs.get(i).getProb());
                System.out.println(labelProbs.get(i).getLabel());
            }

        } catch (Exception e){
            Log.i("TAG", "Some exception " + e);
            e.printStackTrace(System.out);
//            ans.update(0.0f, "Error getPixels" + e);
        }
        return labelProbs;
    }


    public static Bitmap getResizedBitmap(Bitmap image, int newHeight, int newWidth) {
        int width = image.getWidth();
        int height = image.getHeight();
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        // create a matrix for the manipulation
        Matrix matrix = new Matrix();
        // resize the bit map
        matrix.postScale(scaleWidth, scaleHeight);
        // recreate the new Bitmap
        Bitmap resizedBitmap = Bitmap.createBitmap(image, 0, 0, width, height,
                matrix, false);
        return resizedBitmap;
    }



    @Override
    public void close() {
        inferenceInterface.close();
        inferenceInterface = null;
    }
}
