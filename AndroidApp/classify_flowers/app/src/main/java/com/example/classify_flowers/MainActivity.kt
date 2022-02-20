package com.example.classify_flowers

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import com.example.classify_flowers.ml.FlowerClassificationDensenet201Quant
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {

    lateinit var bitmap: Bitmap
    lateinit var imgview: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imgview = findViewById(R.id.imageView)

        val filename = "labels.txt"
        val inputstring = application.assets.open(filename).bufferedReader().use {it.readText() }
        var townlist = inputstring.split("\n")

        var tv:TextView = findViewById(R.id.textView)

        var select: Button = findViewById(R.id.button)

        select.setOnClickListener(View.OnClickListener {

            var intent: Intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "image/*"

            startActivityForResult(intent, 100)
        })

        var predict:Button = findViewById(R.id.button2)
        predict.setOnClickListener(View.OnClickListener {

            var resized: Bitmap = Bitmap.createScaledBitmap(bitmap, 224,224,true)
            val model = FlowerClassificationDensenet201Quant.newInstance(this)
//            // Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)

            var tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(resized)
            var byteBuffer = tensorImage.buffer

            inputFeature0.loadBuffer(byteBuffer)

            // Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer

            var max = getMax(outputFeature0.floatArray)
            tv.setText(townlist[max])
            // Releases model resources if no longer used.
            model.close()
        })

    }
    override fun onActivityResult(requestCode: Int, resultCode: Int, data:Intent?){
        super.onActivityResult(requestCode, resultCode, data)
        imgview.setImageURI(data?.data)
        var uri: Uri?= data?.data

        bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
    }

    fun getMax(arr:FloatArray):Int{

        var ind=0
        var min = 0.0f
        for (i in 0..103)
        {
            if(arr[i]>min)
            {
                ind=i
                min=arr[i]
            }
        }
        if (min < 0.4f)
        {
            ind=104
        }
        return ind

    }
}