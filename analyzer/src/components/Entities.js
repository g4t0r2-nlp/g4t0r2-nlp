import React, { useEffect, useState } from 'react'
import './Entities.css'
import entityList from '../assets/entity_list.json';
import TurkcellLogo from '../assets/turkcell_logo.png';
import TurkTelekomLogo from '../assets/turktelekom_logo.png'
import PaycellLogo from '../assets/paycell_logo.png';
import FizyLogo from '../assets/fizy_logo.jpg';
import DefaultLogo from '../assets/defaultlogo.png';

//<div className='entity-logo'>
//              <img src={turkcell_logo} className='entity-logo' />
//        </div>

const logoPaths = {
    'turkcell': TurkcellLogo,
    'paycell': PaycellLogo,
    'fizy': FizyLogo,
    'turk telekom' : TurkTelekomLogo
};


const Entities = ({ aspect, sentiment }) => {
    
    const [logo, setLogo] = useState('');
    useEffect(() => {
        const entityData = entityList.data;
        const keys = Object.keys(entityData);

        const matchedKey = keys.find(key => aspect.toLowerCase().includes(key.toLowerCase()));

        if (matchedKey) {
            setLogo(logoPaths[entityData[matchedKey]] || DefaultLogo);
        } else {
            setLogo(DefaultLogo);
        }
    })

    return (
        <div className='entity'>
            <div className='entity-logo'>
                <img src={logo} className='entity-logo' />
            </div>
            <div className='entity-name'>
                <h4> {aspect} </h4>
            </div>
            <div className='sentiment'>
                <h4> {sentiment} </h4>
            </div>
        </div>
    )
}

export default Entities