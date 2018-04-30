package com.imageclassification.anqitu.animandroid.Model;

import android.graphics.Bitmap;
import android.graphics.RectF;

import java.util.List;

public interface Classifier {
    /**
     * An immutable result returned by a Classifier describing what was recognized.
     */

    Classification recognizeImage(Bitmap bitmap);

//    void enableStatLogging(final boolean debug);

//    String getStatString();

    void close();
}

