package com.imageclassification.anqitu.animandroid;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.imageclassification.anqitu.animandroid.Model.LabelProb;
import com.imageclassification.anqitu.animandroid.Model.Classifier;
import com.imageclassification.anqitu.animandroid.Model.ImageClassifier;

import java.io.FileNotFoundException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final int INPUT_SIZE = 150;

    private static final String INPUT_NAME = "input_1";
    private static final String OUTPUT_NAME = "dense_2/Softmax";
    private static final String MODEL_FILE = "file:///android_asset/opt_VGG16_animal_ver1.pb";
    private static final String LABEL_FILE = "file:///android_asset/label_ver1.txt";


    private Classifier classifier;

    private Executor executor = Executors.newSingleThreadExecutor();
    private TextView textViewResult;
    private Button btnDetectObject, btnTakePhoto, btnImportPhoto;
    private ImageView photoView, imageView, matchView;
    private LinearLayout photoLayout, resultLayout;

    Bitmap photoBitmap;

    public final static int REQUEST_CAMERA = 1;
    public final static int REQUEST_GALLERY = 2;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        photoLayout = findViewById(R.id.linearLayoutPhoto);
        resultLayout = findViewById(R.id.linearLayoutResult);
        photoView = findViewById(R.id.photoView);
        imageView = findViewById(R.id.imageView);
        matchView = findViewById(R.id.matchView);
        textViewResult = findViewById(R.id.textViewResult);
        btnImportPhoto = findViewById(R.id.btnImportPhoto);
        btnTakePhoto = findViewById(R.id.btnTakePhoto);
        btnDetectObject = findViewById(R.id.btnDetectObject);

        setGetPhotoView();

        addEvents();

        initTensorFlowAndLoadModel();
        int PERMISSION_ALL = 1;
        String[] PERMISSIONS = {Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.CAMERA};
        if(!hasPermissions(this, PERMISSIONS)){
            ActivityCompat.requestPermissions(this, PERMISSIONS, PERMISSION_ALL);
        }
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

        setGetPhotoView();

        super.onActivityResult(requestCode, resultCode, data);
        switch(requestCode){
            case REQUEST_CAMERA:
                if(resultCode == Activity.RESULT_OK) {
                    photoBitmap = handlePhotoFromCamera(data);
                }
                break;
            case REQUEST_GALLERY:
                if(resultCode == Activity.RESULT_OK) {
                    photoBitmap = handlePhotoFromGallery(data);
                }
                break;
        }



        if (photoBitmap != null) {
            photoView.setImageBitmap((photoBitmap));
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
    }

    private void getPictureFromCamera(){
        Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(cameraIntent, REQUEST_CAMERA);
    }

    private void detectObject(){

        setResultView();

        try{
            BitmapDrawable drawable = (BitmapDrawable) photoView.getDrawable();
            Bitmap bitmap = drawable.getBitmap();

            String text = "";

            ArrayList<LabelProb> labelProbs = classifier.recognizeImage(bitmap);


            if (labelProbs == null) {
                text += ": ?\n";
            } else {
                for (int i = 0; i < 3; ++i) {
                    text += String.format("%s (Similarity: %.2f) \n", labelProbs.get(i).getLabel(), labelProbs.get(i).getProb());
                }
            }

            imageView.setImageBitmap((photoBitmap));
            matchView.setImageDrawable(getMatchedImage(labelProbs.get(0).getLabel()));
            textViewResult.setText(text);

        } catch (Exception e) {
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

    public static boolean hasPermissions(Context context, String... permissions) {
        if (context != null && permissions != null) {
            for (String permission : permissions) {
                if (ActivityCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED) {
                    return false;
                }
            }
        }
        return true;
    }

    public void setGetPhotoView() {
        photoLayout.setVisibility(View.VISIBLE);
        resultLayout.setVisibility(View.GONE);
        textViewResult.setVisibility((View.GONE));
    }



    public void setResultView() {
        photoLayout.setVisibility(View.GONE);
        resultLayout.setVisibility(View.VISIBLE);
        textViewResult.setVisibility((View.VISIBLE));
    }

    public Drawable getMatchedImage (String label) {
        switch (label) {
            case "Brave Tiger":
                return getDrawable(R.drawable.tiger);
            case "Curious Cat":
                return getDrawable(R.drawable.cat);
            case "Cute Koala":
                return getDrawable(R.drawable.koala);
            case "Lovely Rabbit":
                return getDrawable(R.drawable.rabbit);
            case "Loyal Dog":
                return getDrawable(R.drawable.dog);
            case "Powerful Lion":
                return getDrawable(R.drawable.lion);
            case "Smart Fox":
                return getDrawable(R.drawable.fox);
        }
        return getDrawable(R.drawable.dog);
    }


}
