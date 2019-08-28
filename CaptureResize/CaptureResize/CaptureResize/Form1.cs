using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using DirectShowLib;
using System.IO;
using System.Runtime.InteropServices;
using Luxand;


namespace CaptureResize
{
    public partial class frmMain : Form
    {
        Capture capture;

        ImageHandler handlerImage;

        static private string pathfolder = "";

        private int size_image = 128;

        private int quality = 100;

        Mat frame;

        Bitmap bmp;

        List<Rectangle> faces;

        public frmMain()
        {
            
            InitializeComponent();
            this.Load += FrmMain_Load;

            txtNameFolder.Text = Properties.Settings.Default.NameFolder;

            if (!String.IsNullOrEmpty(Properties.Settings.Default.NameFolder))
            {
                pathfolder = Application.StartupPath + @"/" + Properties.Settings.Default.NameFolder;
            }
            else
                cboDevice.Enabled = false;

            LoadDevice();
            LoadNumberImage();
        }

        private void FrmMain_Load(object sender, EventArgs e)
        {
            this.WindowState = FormWindowState.Normal;
        }

        private void CreateFolder()
        {

            pathfolder = Application.StartupPath + @"/" + Properties.Settings.Default.NameFolder;
            if (!Directory.Exists(pathfolder))
                Directory.CreateDirectory(pathfolder);
        }

        private void LoadDevice()
        {
            DsDevice[] devices = DsDevice.GetDevicesOfCat(FilterCategory.VideoInputDevice);
            foreach (var item in devices)
                cboDevice.Items.Add(item.Name);
        }

        private void LoadNumberImage()
        {
           // lblNumberImage.Text = "Số lượng hình ảnh trong folder : " + Directory.GetFiles(pathfolder, "*", SearchOption.AllDirectories).Length.ToString();
        }

        private void Capture_ImageGrabbed(object sender, EventArgs e)
        {
            try
            {
                frame = new Mat();
                capture.Retrieve(frame, 0);
                Mat image = frame; //Read the files as an 8-bit Bgr image  
                long detectionTime;
                faces = new List<Rectangle>();

                DetectFace.Detect(
                  image,"haarcascade_frontalface_alt2.xml",
                  faces,
                  out detectionTime);
                foreach (Rectangle face in faces)
                {
                    CvInvoke.Rectangle(image, face, new Bgr(Color.Red).MCvScalar, 1);
                    Bitmap c = frame.Bitmap;
                    bmp = new Bitmap(face.Size.Width, face.Size.Height);
                    Graphics g = Graphics.FromImage(bmp);
                    g.DrawImage(c, 0, 0, face, GraphicsUnit.Pixel);
                }
               
                ptbCamera.Image = frame.Bitmap;

                System.Threading.Thread.Sleep((int)capture.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.Fps));
            }
            catch(Exception ex) {
                MessageBox.Show(ex.Message);
            }
        }

        private void btnCapture_Click(object sender, EventArgs e)
        {
            Bitmap image = bmp;

            ptbCapture.Image = image;

            handlerImage = new ImageHandler();

            handlerImage.Save(image, size_image, size_image, quality,pathfolder + "/" + DateTime.Now.ToString("dd_MM_yyyy_hh_mm_ss") + ".jpg");

            LoadNumberImage();
        }

        private void cboDevice_SelectedIndexChanged(object sender, EventArgs e)
        {
            capture = new Capture(cboDevice.SelectedIndex);
            capture.ImageGrabbed += Capture_ImageGrabbed;
            capture.Start();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Properties.Settings.Default.NameFolder = txtNameFolder.Text;
            Properties.Settings.Default.Save();
            CreateFolder();
            cboDevice.Enabled = true;
        }
    }
}
