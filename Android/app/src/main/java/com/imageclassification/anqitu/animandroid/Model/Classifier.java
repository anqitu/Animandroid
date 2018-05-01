package com.imageclassification.anqitu.animandroid.Model;

import android.graphics.Bitmap;

import java.util.ArrayList;

public interface Classifier {
    /**
     * An immutable result returned by a Classifier describing what was recognized.
     */

    ArrayList<LabelProb> recognizeImage(Bitmap bitmap);

//    void enableStatLogging(final boolean debug);

//    String getStatString();

    void close();
}

