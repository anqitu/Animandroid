package com.imageclassification.anqitu.animandroid;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.imageclassification.anqitu.animandroid.Model.Classification;
import com.imageclassification.anqitu.animandroid.Model.Classifier;
import com.imageclassification.anqitu.animandroid.Model.ImageClassifier;

import java.io.FileNotFoundException;
import java.io.InputStream;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity{

    private static final int INPUT_SIZE = 150;

    private static final String INPUT_NAME = "input_1";
    private static final String OUTPUT_NAME = "dense_2/Softmax";
    private static final String MODEL_FILE = "file:///android_asset/opt_VGG16_animal_ver1.pb";
    private static final String LABEL_FILE = "file:///android_asset/label_ver1.txt";


    private Classifier classifier;

    private Executor executor = Executors.newSingleThreadExecutor();
    private TextView textViewResult;
    private Button btnDetectObject, btnTakePhoto, btnImportPhoto;
    private ImageView imageView;

    public final static int REQUEST_CAMERA = 1;
    public final static int REQUEST_GALLERY = 2;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = findViewById(R.id.imageView);
        textViewResult = findViewById(R.id.textViewResult);
        btnImportPhoto = findViewById(R.id.btnImportPhoto);
        btnTakePhoto = findViewById(R.id.btnTakePhoto);
        btnDetectObject = findViewById(R.id.btnDetectObject);

        addEvents();

        initTensorFlowAndLoadModel();
        makeButtonVisible();
    }

    private void addEvents(){
        btnImportPhoto.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                getPictureFromGallery();
            }
        });

        btnTakePhoto.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                getPictureFromCamera();
            }
        });

        btnDetectObject.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                detectObject();
            }
        });
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent data) {

        super.onActivityResult(requestCode, resultCode, data);
        Bitmap bitmap = null;
        switch(requestCode){
            case REQUEST_CAMERA:
                if(resultCode == Activity.RESULT_OK) {
                    bitmap = handlePhotoFromCamera(data);
                }
                break;
            case REQUEST_GALLERY:
                if(resultCode == Activity.RESULT_OK) {
                    bitmap = handlePhotoFromGallery(data);
                }
                break;
        }



        if (bitmap != null) {
            imageView.setImageBitmap((bitmap));
            System.out.println("imageView is set with bitmap of taken photo");
        } else {
            System.out.println("Null Bitmap");
        }
    }


    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                     classifier = ImageClassifier.create(getAssets(), MODEL_FILE,
                            LABEL_FILE, INPUT_SIZE, INPUT_NAME, OUTPUT_NAME);
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

    private void getPictureFromGallery(){
        Intent galleryImportIntent = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        galleryImportIntent.setType("image/*");
        startActivityForResult(galleryImportIntent, REQUEST_GALLERY);
        textViewResult.setText("");
    }

    private void getPictureFromCamera(){
        Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(cameraIntent, REQUEST_CAMERA);
        textViewResult.setText("");
    }

    private void detectObject(){
        try{
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
        }catch(Exception e){
            Log.i("TAG", "Some exception " + e);
            e.printStackTrace(System.out);
            textViewResult.setText(R.string.error_message);
        }

    }

    private Bitmap handlePhotoFromCamera(Intent data){
        Bitmap takenPictureData = (Bitmap)data.getExtras().get("data");
        return takenPictureData;
    }

    private Bitmap handlePhotoFromGallery(Intent data){
        Bitmap selectedImage = null;
        try {
            Uri imageUri = data.getData();
            InputStream imageStream = getContentResolver().openInputStream(imageUri);
            selectedImage = BitmapFactory.decodeStream(imageStream);
            System.out.println("Photo Imported from Gallery");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.out.println("Photo Importing Failed");
        }
        return selectedImage;
    }
}
