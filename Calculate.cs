using System.Collections;
using System.Diagnostics;
using System.IO;
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Calculate : MonoBehaviour
{
   public Text lblPrint;
   public InputField txtAge, txtHeight, txtWeight, txtBMI;
   public Dropdown ddlGender, ddlLevel;
   public string gender, level;

    // Start is called before the first frame update
    void Start()
    {
        lblPrint  = GameObject.Find("lblPrint").GetComponent<Text>();
        txtAge = GameObject.Find("txtAge").GetComponent<InputField>();
        txtHeight = GameObject.Find("txtHeight").GetComponent<InputField>();
        txtWeight = GameObject.Find("txtWeight").GetComponent<InputField>();
        txtBMI = GameObject.Find("txtBMI").GetComponent<InputField>();
        ddlGender  = GameObject.Find("ddlGender").GetComponent<Dropdown>();
        ddlLevel  = GameObject.Find("ddlLevel").GetComponent<Dropdown>();
    }

    // Update is called once per frame
    void Update()
    {
        gender = ddlGender.options[ddlGender.value].text;
        level = ddlLevel.options[ddlLevel.value].text;
    }
    public void run_cmd(string cmd, string args)
    {
            // string fileName = "/home/akshatha/projects/try.py";

            // Process p = new Process();
            // p.StartInfo = new ProcessStartInfo("/home/akshatha/robot_env/bin/python", fileName)
            // {
            //     RedirectStandardOutput = true,
            //     UseShellExecute = false,
            //     CreateNoWindow = true
            // };
            // p.Start();

            // string output = p.StandardOutput.ReadToEnd();
            // p.WaitForExit();

            // UnityEngine.Debug.Log(output);

            // //Console.ReadLine();
            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = "/home/akshatha/robot_env/bin/python";
            //start.WorkingDirectory = @"D:\script";
            //UnityEngine.Debug.Log(txtAge.text + txtHeight.text+ txtWeight.text+ txtBMI.text+ level+ gender);
            //start.Arguments = string.Format("/home/akshatha/projects/Spine-compression-prediction/main.py --model_name linear_regression");
            start.Arguments = string.Format("/home/akshatha/projects/Spine-compression-prediction/main.py --Age {0} --Heightcm {1} --Weightkg {2} --BMI {3} --level {4} --Gender {5}", txtAge.text, txtHeight.text, txtWeight.text, txtBMI.text, level, gender);
            start.UseShellExecute = false;
            start.RedirectStandardOutput = true;
            using (Process process = Process.Start(start))
            {
                using (StreamReader reader = process.StandardOutput)
                {
                    string result = reader.ReadToEnd();
                    UnityEngine.Debug.Log(result);
                    
                    //lblPrint.text = result;
                }
            }

    }
    public void onCalculate()
    {
        run_cmd("python /home/akshatha/projects/Spine-compression-prediction/main.py", "");
    }
}
