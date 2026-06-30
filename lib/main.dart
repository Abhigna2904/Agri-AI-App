import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:easy_localization/easy_localization.dart';

void main() async {

  WidgetsFlutterBinding.ensureInitialized();

  await EasyLocalization.ensureInitialized();

  runApp(

    EasyLocalization(
      supportedLocales: const [
        Locale('en'),
        Locale('te'),
        Locale('hi'),
      ],
      path: 'assets/translations',
      fallbackLocale: const Locale('en'),
      child: const MyApp(),
    ),
  );
}

class MyApp extends StatelessWidget {

  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {

    return MaterialApp(

      debugShowCheckedModeBanner: false,

      localizationsDelegates: context.localizationDelegates,
      supportedLocales: context.supportedLocales,
      locale: context.locale,

      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {

  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {

  File? _image;

  String disease = "";
  String solution = "";

  final ImagePicker _picker = ImagePicker();

  Future pickImage() async {

    final pickedFile =
        await _picker.pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {

      setState(() {
        _image = File(pickedFile.path);
      });

      predictDisease();
    }
  }

  Future predictDisease() async {

    try {

      print("Sending image to Flask API...");

      print("Using URL: http://10.62.16.79:5000/predict");

      var request = http.MultipartRequest(
        'POST',
        Uri.parse('http://10.62.16.79:5000/predict')
      );

request.fields['language'] =
    context.locale.languageCode;

      request.files.add(
        await http.MultipartFile.fromPath(
          'image',
          _image!.path,
        ),
      );

      var response = await request.send().timeout(
        const Duration(seconds: 60),
      );

      print("Status Code: ${response.statusCode}");

      var responseData = await response.stream.bytesToString();

      print("Response Data: $responseData");

      if (response.statusCode == 200) {

        var jsonData = jsonDecode(responseData);

        setState(() {
          disease = jsonData['disease'];
          solution = jsonData['solution'];
        });

      } else {

        setState(() {
          disease = "API Error";
          solution = responseData;
        });
      }

    } catch (e) {

      print("ERROR: $e");

      setState(() {
        disease = "Connection Failed";
        solution = e.toString();
      });
    }
  }

  @override
  Widget build(BuildContext context) {

    return Scaffold(

      appBar: AppBar(
        title: Text("title".tr()),
        centerTitle: true,

        actions: [

          PopupMenuButton<String>(

            onSelected: (value) {

              if (value == 'en') {
                context.setLocale(const Locale('en'));
              }

              if (value == 'te') {
                context.setLocale(const Locale('te'));
              }

              if (value == 'hi') {
                context.setLocale(const Locale('hi'));
              }
            },

            itemBuilder: (context) => [

              const PopupMenuItem(
                value: 'en',
                child: Text("English"),
              ),

              const PopupMenuItem(
                value: 'te',
                child: Text("తెలుగు"),
              ),

              const PopupMenuItem(
                value: 'hi',
                child: Text("हिन्दी"),
              ),
            ],
          ),
        ],
      ),

      body: SingleChildScrollView(

        child: Center(

          child: Padding(
            padding: const EdgeInsets.all(20),

            child: Column(
              children: [

                const SizedBox(height: 20),

                _image != null
                    ? Image.file(
                        _image!,
                        height: 250,
                      )
                    : const Text(
                        "No Image Selected",
                        style: TextStyle(fontSize: 18),
                      ),

                const SizedBox(height: 20),

                ElevatedButton(
                  onPressed: pickImage,
                  child: Text("upload".tr()),
                ),

                const SizedBox(height: 10),

                ElevatedButton.icon(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red,
                    foregroundColor: Colors.white,
                  ),
                  onPressed: () {
                    setState(() {
                      _image = null;
                      disease = "";
                      solution = "";
                    });
                  },
                  icon: const Icon(Icons.clear),
                  label: const Text("Clear"),
                ),

                const SizedBox(height: 30),

                Text(
                  disease,
                  style: const TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                    color: Colors.red,
                  ),
                  textAlign: TextAlign.center,
                ),

                const SizedBox(height: 20),

                Text(
                  solution,
                  style: const TextStyle(
                    fontSize: 18,
                  ),
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}