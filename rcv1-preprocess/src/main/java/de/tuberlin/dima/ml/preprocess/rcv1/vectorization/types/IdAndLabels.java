package de.tuberlin.dima.ml.preprocess.rcv1.vectorization.types;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;

import com.google.common.primitives.Ints;

/**
 * For categorical labels, that were exploded to binary labels Each binary label is assumed to be
 * either 0 or 1
 * 
 * A mapping from label-id to label-name can be described using DatasetInfo
 */
public class IdAndLabels implements WritableComparable<IdAndLabels> {

    private int id;
    private Vector labels;

    public void set(int id, Vector labels) {
        this.id = id;
        this.labels = labels;
    }

    public int getId() {
        return this.id;
    }

    public Vector getLabels() {
        return this.labels;
    }

    @Override
    public int compareTo(IdAndLabels o) {
        return Ints.compare(this.id, o.id);
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(this.id);
        out.writeInt(this.labels.size());

        for (Element e : this.labels.nonZeroes()) {
            out.writeDouble(e.get());
        }
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        this.id = in.readInt();
        int numCodes = in.readInt();

        this.labels = new DenseVector(numCodes);
        for (int index = 0; index < numCodes; index++) {
            this.labels.set(index, in.readDouble());
        }
    }
}