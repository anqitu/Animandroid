package com.imageclassification.anqitu.animandroid;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.imageclassification.anqitu.animandroid.Model.Classification;
import com.imageclassification.anqitu.animandroid.Model.Classifier;
import com.imageclassification.anqitu.animandroid.Model.ImageClassifier;

import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {

    private static final int INPUT_SIZE = 150;
//    private static final int IMAGE_MEAN = 117;
//    private static final float IMAGE_STD = 1;

    private static final String INPUT_NAME = "input_1";
    private static final String OUTPUT_NAME = "dense_2/Softmax";
    private static final String MODEL_FILE = "file:///android_asset/opt_VGG16_animal_ver1.pb";
    private static final String LABEL_FILE = "file:///android_asset/label_ver1.txt";

//    private static final String INPUT_NAME = "conv2d_1_input";
//    private static final String OUTPUT_NAME = "dense_2/Softmax";
//    private static final String MODEL_FILE = "file:///android_asset/opt_mnist_convnet.pb";
//    private static final String LABEL_FILE = "file:///android_asset/labels.txt";

    private Classifier classifier;

    private Executor executor = Executors.newSingleThreadExecutor();
    private TextView textViewResult;
    private Button btnDetectObject, btnTakePhoto;
    private ImageView imageView;




    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = findViewById(R.id.imageView);
        textViewResult = findViewById(R.id.textViewResult);
        btnTakePhoto = findViewById(R.id.btnTakePhoto);
        btnDetectObject = findViewById(R.id.btnDetectObject);

        btnTakePhoto.setOnClickListener(this);
        btnDetectObject.setOnClickListener(this);

        initTensorFlowAndLoadModel();
        makeButtonVisible();
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent data) {

        super.onActivityResult(requestCode, resultCode, data);
        Bitmap bitmap = (Bitmap)data.getExtras().get("data");
        imageView.setImageBitmap((bitmap));
        System.out.println("imageView is set with bitmap of taken photo");
    }



    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                     classifier = ImageClassifier.create(getAssets(), MODEL_FILE,
                            LABEL_FILE, INPUT_SIZE, INPUT_NAME, OUTPUT_NAME);
//                            LABEL_FILE, INPUT_SIZE, IMAGE_MEAN, IMAGE_STD, INPUT_NAME, OUTPUT_NAME);
                    makeButtonVisible();
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing Model!", e);
                }
            }
        });
    }

    private void makeButtonVisible() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                btnDetectObject.setVisibility(View.VISIBLE);
            }
        });
    }

    @Override
    public void onClick(View view) {
        if (view.getId() == R.id.btnTakePhoto) {
            Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
            startActivityForResult(cameraIntent, 0);
            textViewResult.setText("");

        } else if (view.getId() == R.id.btnDetectObject) {
            BitmapDrawable drawable = (BitmapDrawable) imageView.getDrawable();
            Bitmap bitmap = drawable.getBitmap();

            String text = "";

            final Classification res = classifier.recognizeImage(bitmap);
            if (res.getLabel() == null) {
                text += ": ?\n";
            } else {
                text += String.format("%s (Similarity: %f)", res.getLabel(), res.getConf());
            }
            textViewResult.setText(text);
        }

    }
}
