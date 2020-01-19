import React from 'react';
import { useState, useEffect } from 'react';
import MapView, { Marker } from 'react-native-maps';
import { StyleSheet, Text, View, Dimensions, Button, Alert } from 'react-native';
import Constants from 'expo-constants';
import * as Location from 'expo-location';
import * as Permissions from 'expo-permissions';
import { Camera } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';

let IP = 'http://10.200.28.73:3000/';
var text = "";

export default class App extends React.Component {
  ;

  state = {
    location: 0,
    errorMessage: null,
    latitude: 1,
    longitude: 1,
    canTakePicture: true,
    markerPos: {
      "latitude": 1,
      "longitude": 1
    },
    photo: null,
    type: Camera.Constants.Type.back
  };

  UNSAFE_componentWillMount() {
    if (Platform.OS === 'android' && !Constants.isDevice) {
      this.setState({
        errorMessage: 'Oops, this will not work on Sketch in an Android emulator. Try it on your device!',
      });
    } else {
      this._getLocationAsync();
    }
  }

  _getLocationAsync = async () => {
    let { status } = await Permissions.askAsync(Permissions.LOCATION);
    if (status !== 'granted') {
      this.setState({
        errorMessage: 'Permission to access location was denied',
      });
    }

    let location = await Location.getCurrentPositionAsync({});
    let latitude = location.coords.latitude
    let longitude = location.coords.longitude

    this.setState({ location })
    this.setState({ latitude })
    this.setState({ longitude })

    let markerPos = {
      "latitude": latitude,
      "longitude": longitude
    }
    this.setState({ markerPos })
  };

  state = {
    hasPermission: null,
    type: Camera.Constants.Type.back,
  }

  async componentDidMount() {
    const { status } = await Permissions.askAsync(Permissions.CAMERA);
    this.setState({ hasPermission: status === 'granted' });
  }

  async sendCoordinates() {
    try {
      let response = await fetch(
        IP + 'lat/' + this.state.latitude.toString() + '/lng/' + this.state.longitude.toString(),
      );
      let responseJson = await response.json();
      return responseJson;
    } catch (error) {
      console.error(error);
    }
  }

  async takePicture() {
    if (this.camera) {
      let photo = await this.camera.takePictureAsync();
      console.log(photo);

      const data = new FormData();
      data.append('name', 'avatar');
      data.append('fileData', {
        uri: photo.uri,
        type: 'image/jpeg',
        name: 'image'
      });
      console.log(data)
      const config = {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'multipart/form-data',
        },
        body: data,
      };
      fetch(IP + "upload", config)
        .then((checkStatusAndGetJSONResponse) => {
          console.log('Done')
        }).catch((err) => { console.log('oup') });
    }
  }

  render() {
    var hasAccessToCamera = false

    const { hasPermission } = this.state
    if (hasPermission === null) {
      return <View />;
    } else if (hasPermission === false) {
      return <Text>No access to camera</Text>;
    } else {
      hasAccessToCamera = true
    }


    if (this.state.errorMessage) {
      text = this.state.errorMessage;
    } else if (this.state.location) {
    }



    return (


      <View style={styles.container}>
        <View style={this.state.canTakePicture ? styles.camera : styles.hidden}>

          <Camera
            style={{ flex: 1 }}
            type={this.state.cameraType}
            ref={ref => {
              this.camera = ref;
            }}
          >
          </Camera>
          <Button style={styles.buttonStyle}
            title="Take picture"
            onPress={() => {

              console.log(this.state.markerPos)
              this.sendCoordinates()
              this.takePicture()

              Alert.alert('Done')

            }}
          />
          <Button style={styles.buttonStyle}
            title="Cancel"
            onPress={() => {
              // Alert.alert('Simple Button pressed')
              let canTakePicture = false
              this.setState({ canTakePicture });
              this._getLocationAsync();
            }}
          />
        </View>



        <MapView
          style={!this.state.canTakePicture ? styles.mapStyle : styles.hidden}
          initialRegion={{
            latitude: 1,
            longitude: 1,
            latitudeDelta: 0.001,
            longitudeDelta: 0.001
          }}
          region={{
            latitude: this.state.latitude != null ? this.state.latitude : 1,
            longitude: this.state.longitude != null ? this.state.longitude : 1,
            latitudeDelta: 0.001,
            longitudeDelta: 0.001
          }}
        >
          <MapView.Marker
            // draggable
            // onDragEnd={(e) => this.setState({ markerPos: e.nativeEvent.coordinate })}
            coordinate={{
              latitude: this.state.latitude != null ? this.state.latitude : 1,
              longitude: this.state.longitude != null ? this.state.longitude : 1
            }}
          />
        </MapView>
        <View style={!this.state.canTakePicture ? styles.buttonContainer : styles.hidden}>
          <Button style={styles.buttonStyle}
            title="Identify bin"
            onPress={() => {
              // Alert.alert('Simple Button pressed')
              let currentCameraState = this.state.canTakePicture
              let newCameraState = !currentCameraState
              let canTakePicture = newCameraState
              this.setState({ canTakePicture });
            }}


          />
          <Button style={styles.buttonStyle}
            title="Show current location"
            onPress={() => {
              this._getLocationAsync();
            }}

          />
        </View>

      </View>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttonContainer: {
    width: Dimensions.get('window').width
  },
  buttonStyle: {
    backgroundColor: '#ffffff'

  },
  mapStyle: {
    width: Dimensions.get('window').width,
    // height: Dimensions.get('window').height - 200,
    flex: 1
  },
  hidden: {
    width: 0,
    height: 0,
  },
  camera: {
    width: Dimensions.get('window').width,
    // height: Dimensions.get('window').height - 100
    flex: 1
  }
});