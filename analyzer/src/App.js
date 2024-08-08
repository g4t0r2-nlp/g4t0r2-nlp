import gator_logo from './assets/gator_logo.jpeg'
import './App.css';
import { useState } from 'react';
import Entities from './components/Entities';
import axios from "axios"
import React from 'react'
import background from './assets/crystal_background.jpg'


function App() {

  const [comment, setComment] = useState("");
  const [showEntityList, setShowEntityList] = useState(false);
  const [entities, setEntities] = useState([]);

  // .then(response => response.data["results"].map(e => ({
  //  aspect : e.aspect,
  //  sentiment : e.sentiment
  //})), console.log(response.data["results"][0]["aspect"]))

  const analyzeComment = async () => {
    const data = {
      "text": comment
    }
    await axios.post("https://thealper2-aspect-sentiment-pipeline.hf.space/predict/", data)
      .then(response => setEntities(response.data["results"].map(e => ({
        aspect: e.aspect,
        sentiment: e.sentiment
      }))))
      .catch(err => alert(err));
    setShowEntityList(true);
  }

  return (
    <div className="App" style={{ backgroundImage: `url(${background})` }}>
      <div className='header'>
        <h2> Comment Analyzer </h2>
        <img src={gator_logo} className='gator-logo' />
      </div>
      <div className='input'>
        <textarea
          className='custom-input'
          value={comment}
          onChange={(e) => setComment(e.target.value)} />
        <button className='button' onClick={analyzeComment} >
          Analyze
        </button>
      </div>
      {showEntityList &&
        <div className='entity-list'>
          <div className='row-headers'>
            <h3> Logo </h3>
            <h3> Varlık </h3>
            <h3> Duygu </h3>
          </div>
          {
            entities.map((entity, index) => (
              <React.Fragment key={index}>
                <Entities
                  key={index}
                  aspect={entity.aspect}
                  sentiment={entity.sentiment}
                />
                {index < entities.length - 1 && <hr/>}
              </React.Fragment>
            ))}
        </div>
      }
    </div>
  );
}

export default App;
