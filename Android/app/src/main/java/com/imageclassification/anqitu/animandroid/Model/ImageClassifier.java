package com.imageclassification.anqitu.animandroid.Model;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Trace;
import android.util.Log;

import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
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
    private float[] outputs;
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
     * @param imageMean The assumed mean of the image values.
     * @param imageStd The assumed std of the image values.
     * @param inputName The label of the image input node.
     * @param outputName The label of the output node.
     * @throws IOException
     */
    public static Classifier create(
            AssetManager assetManager,
            String modelFilename,
            String labelFilename,
            int inputSize,
            int imageMean,
            float imageStd,
            String inputName,
            String outputName) {
        ImageClassifier c = new ImageClassifier();
        c.inputName = inputName;
        c.outputName = outputName;

        // Read the label names into memory.
        // TODO(andrewharp): make this handle non-assets.
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

        // Ideally, inputSize could have been retrieved from the shape of the input operation.  Alas,
        // the placeholder node for input in the graphdef typically used does not specify a shape, so it
        // must be passed in as a parameter.
        c.inputSize = inputSize;
//        c.imageMean = imageMean;
//        c.imageStd = imageStd;

        // Pre-allocate buffers.
        c.outputNames = new String[] {outputName};
        c.intValues = new int[inputSize * inputSize];
        c.pixels = new float[inputSize * inputSize * 3];
        c.outputs = new float[numClasses];

        return c;
    }


    @Override
    public Classification recognizeImage(Bitmap bitmap) {
//        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
////        byte[][] result = new byte[1][labels.size()];
//////        inferenceInterface.run(byteBuffer, result);
//////        return getSortedResult(result);
////        return null;
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
//            floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
//            floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
//            floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;

            pixels[i * 3 + 0] = ((val >> 16) & 0xFF) / 255;
            pixels[i * 3 + 1] = ((val >> 8) & 0xFF) / 255;
            pixels[i * 3 + 2] = (val & 0xFF) / 255;
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
        inferenceInterface.fetch(outputName, outputs);
        Trace.endSection();


        Classification ans = new Classification();
        for (int i = 0; i < outputs.length; ++i) {
            System.out.println(outputs[i]);
            System.out.println(labels.get(i));
            if (outputs[i] > ans.getConf()) {
                ans.update(outputs[i], labels.get(i));
            }
        }
        return ans;
    }


    @Override
    public void close() {
        inferenceInterface.close();
//        inferenceInterface = null;
    }

//    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
//        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE);
//        byteBuffer.order(ByteOrder.nativeOrder());
//        int[] intValues = new int[inputSize * inputSize];
//        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
//        int pixel = 0;
//        for (int i = 0; i < inputSize; ++i) {
//            for (int j = 0; j < inputSize; ++j) {
//                final int val = intValues[pixel++];
//                byteBuffer.put((byte) ((val >> 16) & 0xFF));
//                byteBuffer.put((byte) ((val >> 8) & 0xFF));
//                byteBuffer.put((byte) (val & 0xFF));
//            }
//        }
//        return byteBuffer;
//    }

//    private List<Recognition> getSortedResult(byte[][] labelProbArray) {
//
//        PriorityQueue<Recognition> pq =
//                new PriorityQueue<>(
//                        MAX_RESULTS,
//                        new Comparator<Recognition>() {
//                            @Override
//                            public int compare(Recognition lhs, Recognition rhs) {
//                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
//                            }
//                        });
//
//        for (int i = 0; i < labels.size(); ++i) {
//            float confidence = (labelProbArray[0][i] & 0xff) / 255.0f;
//            if (confidence > THRESHOLD) {
//                pq.add(new Recognition("" + i,
//                        labels.size() > i ? labels.get(i) : "unknown",
//                        confidence));
//            }
//        }
//
//        final ArrayList<Recognition> recognitions = new ArrayList<>();
//        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
//        for (int i = 0; i < recognitionsSize; ++i) {
//            recognitions.add(pq.poll());
//        }
//
//        return recognitions;
//    }
}
