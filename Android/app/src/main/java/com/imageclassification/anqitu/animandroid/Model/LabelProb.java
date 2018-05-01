package com.imageclassification.anqitu.animandroid.Model;

public class LabelProb {

    private float prob;
    private String label;

    LabelProb() {
        this.prob = -1.0F;
        this.label = null;
    }

    LabelProb(float prob, String label) {
        this.prob = prob;
        this.label = label;
    }

    void update(float conf, String label) {
        this.prob = conf;
        this.label = label;
    }

    public String getLabel() {
        return label;
    }

    public float getProb() {
        return prob;
    }


//    @Override
//    public int compareTo(@NonNull LabelProb labelProbability2) {
//        if (this.getProb() > labelProbability2.getProb()) return 1;
//        else return 0;
//    }
}